"""One recipe interface and export contract for all conditional terminal GANs."""
from pathlib import Path
import shutil
import numpy as np
import torch
import yaml
from learning.common.environment import environment
from learning.common.io import collection_path, sha256, write_json, write_yaml
from .config import load_config
from .data import prepare, selection
from .engine import train_experiment
from .models import MODELS
from .references import load_reference

TRAINING = dict(seed=1701, epochs=53, batch_size=2000, critic_steps=5,
    generator_learning_rate=1e-4, critic_learning_rate=1e-4, adam_betas=[.5,.9],
    scheduler='plateau', plateau_patience=2, plateau_factor=.5,
    minimum_learning_rate=1e-6, validation_min_delta=.001,
    early_stopping_patience=6, minimum_epochs=9, ema_decay=.99,
    validation_interval=1, log_interval=50, deterministic=True, num_workers=0, device='cuda')

ENTRYPOINT = '''"""Train this GAN using the common terminal-distribution pipeline."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "learning/gan").is_dir())
sys.path.insert(0, str(ROOT))
from learning.gan.train import recipe_main

if __name__ == "__main__":
    recipe_main(Path(__file__).with_name("recipe.yaml"))
'''


def make_recipe(model, dataset_id, *, dataset_name='samples_02', paths_per_parameter=1):
    spec = MODELS[model]
    relative = Path(spec.path)/dataset_name
    leaf = Path('catalog/gan')/relative
    output = Path('gan')/relative
    return dict(schema_version=1, kind='terminal_gan', id=model+'__'+dataset_id+'__gan_256',model=model,
        training=str(leaf/'training.py'),
        dataset=dict(id=dataset_id,path=f'datasets/samples/{spec.path}/{dataset_name}.json'),
        inputs=dict(conditions=list(spec.conditions),fixed=spec.fixed,
                    time_convention={'days_per_year':252,'T':'maturity_days / 252'}),
        observable=dict(name=(list(spec.observable) if isinstance(spec.observable,tuple) else spec.observable),transform=({'name':'log_shift','offset':1e-12} if spec.nonnegative else {'name':'identity'})),
        reference=dict(path=str(output/'reference.npz'),
            conditions_dataset=dict(id='samples_01',path=f'datasets/samples/{spec.path}/samples_01.json'),
            groups_per_split=128, paths_per_condition=8192, selection='random_parameter_groups_disjoint_from_training', selection_seed=2026112091,
            simulation_seed=2026112091827,method=spec.method),
        data=dict(training_rows='exclude_reference_parameters' if paths_per_parameter > 1 else 'all',
                  normalization='training_only'),
        network=dict(hidden_sizes=[256]*3,activation='softplus',softplus_beta=10,
                     embedding_size=150,latent_dimension=5,latent_distribution='uniform_0_1'),
        objective='logistic',training_options=dict(TRAINING),
        evaluation=dict(validation_seed=941,test_seed=942,select='validation',
                        score='mean(W1 / reference_std + 0.1 * abs(log(generated_std / reference_std)))'),
        output=dict(model=str(output/'model.pt'),metrics=str(output/'metrics.json')))


def configuration(recipe, data):
    spec = MODELS[recipe['model']]
    network=recipe['network']
    if (network['hidden_sizes'] != [network['hidden_sizes'][0]]*len(network['hidden_sizes'])
            or network['activation']!='softplus' or network['softplus_beta']!=10
            or network['latent_distribution']!='uniform_0_1' or recipe['objective']!='logistic'
            or recipe['data'] not in ({'training_rows':'all','normalization':'training_only'},
                                    {'training_rows':'exclude_reference_parameters','normalization':'training_only'})):
        raise ValueError('Recipe does not describe the qualified common GAN protocol')
    shared=dict(name='conditional_embedded_mlp',hidden_size=network['hidden_sizes'][0],
                depth=len(network['hidden_sizes'])+1,activation='softplus',
                parameter_embedding_size=network['embedding_size'])
    return load_config(None,dict(dataset=str(data.directory),
        data=dict(observables=list(spec.observables),observable_transforms={name:recipe['observable']['transform'] for name in spec.observables}),
        generator=dict(**shared,latent_dimension=network['latent_dimension'],noise_distribution='uniform',
                       noise_minimum=0.,noise_maximum=1.,noise_embedding_size=network['embedding_size']),
        critic=dict(**shared,observation_embedding_size=network['embedding_size']),
        objective={'name':'logistic'},training=recipe['training_options'],
        evaluation=dict(max_samples=recipe['reference']['groups_per_split']*recipe['reference']['paths_per_condition'],
            mmd_samples=512,projection_count=4,condition_projection_count=4,grouped_conditions=True,
            seed=recipe['evaluation']['validation_seed'],evaluate_test=False,
            call_strikes=[.7,.85,1,1.15,1.3] if spec.observable=='spot' else [])))


def evaluate_checkpoint(recipe, data, run_directory, *, split=2):
    from learning.common.data.terminal_torch import TerminalBatchLoader
    from learning.common.data.terminal_transforms import TerminalTransform
    from .networks import build_generator
    from .metrics import evaluate_generator
    spec=MODELS[recipe['model']]
    config=configuration(recipe,data)
    requested=config['training']['device']
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else requested)
    checkpoint=torch.load(run_directory/'best_model.pt',map_location='cpu',weights_only=False)
    if tuple(checkpoint['tensor_schema']['conditioning_names']) != data.schema.conditioning_names:
        raise ValueError('Checkpoint input coordinates differ from prepared data')
    generator=build_generator(config['generator'],condition_dimension=len(data.schema.conditioning_names),
                              observation_dimension=len(spec.observables)).to(device).eval()
    generator.load_state_dict(checkpoint['ema_generator_state_dict'])
    transform=TerminalTransform.from_dict(checkpoint['transform'])
    loader=TerminalBatchLoader(data,np.flatnonzero(data.split_codes==split),batch_size=8192,shuffle=False,seed=0)
    evaluation={**config['evaluation'],'seed':recipe['evaluation']['test_seed'] if split==2 else recipe['evaluation']['validation_seed']}
    with torch.inference_mode():
        return evaluate_generator(generator,loader,data.schema,transform,device,evaluation)


def export(recipe_path, root, data, run_directory, *, test_metrics, output_root=None, technical=None):
    """Export an already completed run; never retrain or change its checkpoint."""
    import json
    from .metrics import grouped_distribution_metrics
    recipe=yaml.safe_load(recipe_path.read_text())
    run=json.loads((run_directory/'run.json').read_text())
    state=json.loads((run_directory/'state.json').read_text())
    if state['status']!='completed': raise ValueError('Only completed GANs can be exported')
    model=collection_path(root,recipe['output']['model'],'gan',output=True,output_root=output_root)
    model.parent.mkdir(parents=True,exist_ok=True)
    temporary=model.with_suffix('.tmp')
    shutil.copyfile(run_directory/'best_model.pt',temporary)
    temporary.replace(model)
    refpath=collection_path(root,recipe['reference']['path'],'gan')
    conditions,values,meta=load_reference(refpath,recipe['model'])
    g,p=meta['groups_per_split'],meta['paths_per_condition']
    noise=grouped_distribution_metrics(np.repeat(conditions[:g],p,axis=0),
                                      values[:g,:p].reshape(-1,len(MODELS[recipe['model']].observables)),values[:g,p:].reshape(-1,len(MODELS[recipe['model']].observables)))
    metrics=collection_path(root,recipe['output']['metrics'],'gan',output=True,output_root=output_root)
    write_json(metrics,dict(validation=run['metrics']['validation'],test=test_metrics,
        validation_simulator_vs_simulator=noise,selected_epoch=run['best_epoch'],
        selection=selection(data).manifest(),parameter_counts=run['runtime']['parameter_counts']))
    receipt=dict(schema_version=1,status='complete',inputs={k:data.manifest['identity'][k] for k in ('dataset','reference')},
                 output={name:dict(path=recipe['output'][name],sha256=sha256(path))
                         for name,path in [('model',model),('metrics',metrics)]},**(technical or environment(device=run['runtime']['device'])))
    write_yaml(recipe_path.with_name('generation.yaml'),receipt)
    return model


def run_recipe(recipe_path, root, *, cache_root=None, output_root=None, resume=False):
    recipe=yaml.safe_load(recipe_path.read_text())
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(.65)
    print('Training: preparing', recipe['id'], flush=True)
    data=prepare(root,recipe,cache_root=cache_root)
    print('Training: data checks', data.manifest['checks'], flush=True)
    print('Training: starting', recipe['id'], flush=True)
    config=configuration(recipe,data)
    run=(Path(output_root) if output_root else root/'work/learning/gan')/'runs'/recipe['id']
    result=train_experiment(data,selection(data),config,run,resume=resume)
    test=evaluate_checkpoint(recipe,data,run)
    export(recipe_path,root,data,run,test_metrics=test,output_root=output_root)
    return result
