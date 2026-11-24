"""Public GAN recipe contract: leakage, transforms, export and portable entry point."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import yaml
from learning.common.io import collection_path, sha256, write_yaml
from learning.gan.data import prepare, selection
from learning.gan.recipes import make_recipe, run_recipe, ENTRYPOINT
from learning.gan.references import save_reference, load_reference
from learning.gan.models import MODELS
from tools.datasets.metadata_schemas import catalogue_documents

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def tiny(tmp_path):
    recipe=make_recipe('cir','samples_02')
    recipe['reference'].update(groups_per_split=2,paths_per_condition=16)
    recipe['network'].update(hidden_sizes=[8],embedding_size=4,latent_dimension=2)
    recipe['training_options'].update(epochs=1,minimum_epochs=0,batch_size=4,critic_steps=1,device='cpu')
    path=tmp_path/Path(recipe['training']).with_name('recipe.yaml')
    write_yaml(path,recipe)
    dataset=collection_path(tmp_path,recipe['dataset']['path'],'datasets',output=True)
    dataset.parent.mkdir(parents=True)
    rows=[dict(id=i,parameters=dict(mean_reversion=0.5+i/10,long_term_mean=.04,
          volatility=.1,initial_state=.03),maturity_days=126,values={'state':.02+i*.001}) for i in range(8)]
    header={'database_id':'samples_02','model_family':'CIR','row_count':8,'time_convention':{'days_per_year':252}}
    prefix=json.dumps(header,indent=2)[:-2]
    dataset.write_text(prefix+',\n  "results": [\n'+',\n'.join(json.dumps(r) for r in rows)+'\n]\n}\n')
    write_yaml(dataset.with_suffix('.recipe.yaml'),{'shape':{'parameter_count':8,'paths_per_parameter':1,'row_order':'parameter-major, then path-major'}})
    conditions=np.array([[10+i,.04,.1,.03,.5] for i in range(4)],np.float32)
    values=np.array([[.03+j*.001+i*.002 for j in range(32)] for i in range(4)],np.float32)
    reference=collection_path(tmp_path,recipe['reference']['path'],'gan',output=True)
    metadata=dict(model='cir',groups_per_split=2,paths_per_condition=16,
                  condition_names=list(MODELS['cir'].conditions),observable='state',simulation_seed=2026112091827)
    save_reference(reference,conditions,values,metadata)
    return tmp_path,path,recipe,reference


def test_preparation_preserves_all_train_rows_and_grouped_holdouts(tiny):
    root,path,recipe,ref=tiny
    data=prepare(root,recipe)
    selected=selection(data)
    assert selected.manifest()['counts']=={'train':8,'validation':32,'test':32}
    assert data.conditions[:8,0].tolist()==pytest.approx([.5+i/10 for i in range(8)])
    c,v,_=load_reference(ref,'cir')
    np.testing.assert_array_equal(data.conditions[8:],np.repeat(c,16,axis=0))
    np.testing.assert_array_equal(data.observations[8:,0],v[:,:16].ravel())
    assert prepare(root,recipe).fingerprint==data.fingerprint


def test_overlap_ignores_maturity_and_fixed_inputs_are_checked(tiny):
    root,path,recipe,ref=tiny
    c,v,meta=load_reference(ref,'cir')
    c[0]=[.5,.04,.1,.03,1.0]  # Same model as training, different maturity.
    save_reference(ref,c,v,meta)
    with pytest.raises(ValueError,match='overlap'):prepare(root,recipe)
    c[0,0]=10
    save_reference(ref,c,v,meta)
    recipe['inputs']['fixed']={'mean_reversion':999}
    with pytest.raises(ValueError,match='actually varies'):prepare(root,recipe)


def test_training_exports_inference_model_metrics_and_receipt(tiny):
    root,path,recipe,ref=tiny
    run_recipe(path,root)
    receipt=yaml.safe_load(path.with_name('generation.yaml').read_text())
    assert list(receipt)==['schema_version','status','inputs','output','hardware','software']
    assert receipt['hardware']['device']=='cpu'
    for role,entry in receipt['inputs'].items():
        collection='datasets' if role=='dataset' else 'gan'
        assert sha256(collection_path(root,entry['path'],collection))==entry['sha256']
    for entry in receipt['output'].values():
        assert sha256(collection_path(root,entry['path'],'gan'))==entry['sha256']
    import torch
    model=collection_path(root,recipe['output']['model'],'gan')
    checkpoint=torch.load(model,map_location='cpu',weights_only=False)
    assert checkpoint['tensor_schema']['conditioning_names']==list(MODELS['cir'].conditions)
    assert checkpoint['transform']['condition_mean'][0]==pytest.approx(.85)  # train only
    metrics=json.loads(collection_path(root,recipe['output']['metrics'],'gan').read_text())
    assert metrics['test']['sample_count']==32 and metrics['test']['grouped_condition_count']==2
    assert model.is_relative_to(root/'GANS/unpublished')
    assert not (root/'gan').exists()


def test_catalogue_entrypoints_and_dataset_scanner(tmp_path):
    manifest=json.loads((ROOT/'catalog/gan/manifest.json').read_text())
    assert len(manifest['networks'])>=4
    for row in manifest['networks']:
        path=ROOT/row['recipe'];recipe=yaml.safe_load(path.read_text())
        assert path.with_name('training.py').read_text()==ENTRYPOINT
        assert ROOT/recipe['training']==path.with_name('training.py')
        assert 'work/' not in path.read_text()
        assert recipe['dataset']['id']=='samples_02'
        assert recipe['data']['training_rows']=='all'
        assert recipe['reference']['selection']=='stored_in_archive'
        assert 'conditions_dataset' not in recipe['reference']
    leaf=ROOT/manifest['networks'][0]['recipe']
    process=subprocess.run([sys.executable,str(leaf.with_name('training.py')),'--help'],cwd=tmp_path,
                            capture_output=True,text=True,timeout=60)
    assert process.returncode==0,process.stderr
    assert '--resume' in process.stdout and '--prepare-only' in process.stdout
    assert not any('gan' in p.relative_to(ROOT/'catalog').parts for _,p in catalogue_documents(ROOT))


@pytest.mark.parametrize('bad',['/tmp/model.pt','../model.pt','gan','datasets/model.pt'])
def test_collection_rejects_invalid_output_paths(tmp_path,bad):
    with pytest.raises(ValueError):collection_path(tmp_path,bad,'gan',output=True)


@pytest.mark.parametrize('model',['vasicek','g2'])
def test_signed_and_joint_observables_use_same_training_engine(tmp_path,model):
    spec=MODELS[model]
    recipe=make_recipe(model,'samples_02')
    assert recipe['observable']['transform']=={'name':'identity'}
    recipe['reference'].update(groups_per_split=2,paths_per_condition=16)
    recipe['network'].update(hidden_sizes=[8],embedding_size=4,latent_dimension=2)
    recipe['training_options'].update(epochs=1,minimum_epochs=0,batch_size=4,critic_steps=1,device='cpu')
    path=tmp_path/Path(recipe['training']).with_name('recipe.yaml');write_yaml(path,recipe)
    dataset=collection_path(tmp_path,recipe['dataset']['path'],'datasets',output=True)
    dataset.parent.mkdir(parents=True)
    rows=[dict(id=i,parameters={name:(i+1)*.1 for name in spec.conditions[:-1]},
               maturity_days=126,values={name:(i-4)*.01 for name in spec.observables}) for i in range(8)]
    header=dict(database_id='samples_02',model_family=model,row_count=8,time_convention={'days_per_year':252})
    dataset.write_text(json.dumps(header,indent=2)[:-2]+',\n  "results": [\n'+',\n'.join(json.dumps(r) for r in rows)+'\n]\n}\n')
    write_yaml(dataset.with_suffix('.recipe.yaml'),{'shape':{'parameter_count':8,'paths_per_parameter':1,'row_order':'parameter-major, then path-major'}})
    c=np.array([[10+i]* (len(spec.conditions)-1)+[.5] for i in range(4)],np.float32)
    v=np.random.default_rng(10).normal(0,.02,(4,32,len(spec.observables))).astype(np.float32)
    if len(spec.observables)==1:v=v[:,:,0]
    ref=collection_path(tmp_path,recipe['reference']['path'],'gan',output=True)
    save_reference(ref,c,v,dict(model=model,groups_per_split=2,paths_per_condition=16,
        condition_names=list(spec.conditions),observable=spec.observable,simulation_seed=2026112091827))
    data=prepare(tmp_path,recipe)
    assert data.observations.shape==(72,len(spec.observables))
    assert data.observations.min()<0
    run_recipe(path,tmp_path)
    import torch
    checkpoint=torch.load(collection_path(tmp_path,recipe['output']['model'],'gan'),weights_only=False)
    assert len(checkpoint['tensor_schema']['observable_names'])==len(spec.observables)
    assert all(x['name']=='identity' for x in checkpoint['transform']['observation_specs'])


def test_varying_spot_is_an_input_and_curve_models_are_excluded():
    for name in ('cev','sabr','rough_sabr'):
        recipe=make_recipe(name,'samples_02')
        assert 'spot' in recipe['inputs']['conditions']
        assert 'spot' not in recipe['inputs']['fixed']
    assert not {'hull_white','cir_plus_plus','g2_plus_plus'} & MODELS.keys()


def test_joint_evaluation_detects_wrong_dependence_with_identical_marginals():
    from learning.gan.metrics import grouped_distribution_metrics
    x=np.linspace(-1,1,64)
    real=np.column_stack((x,x))
    fake=np.column_stack((x,-x))
    metrics=grouped_distribution_metrics(np.zeros((64,1)),real,fake)
    assert metrics['grouped_scaled_w1_mean']==pytest.approx(0)
    assert metrics['grouped_correlation_mae']==pytest.approx(2)


def test_grouped_samples_reserve_whole_packs_and_reuse_iid_holdouts(tiny):
    """A different maturity cannot leak a held-out model tuple through its pack."""
    root,path,recipe,ref=tiny
    dataset=collection_path(root,recipe['dataset']['path'],'datasets')
    payload=json.loads(dataset.read_text())
    for row in payload['results']:
        row['parameters']['mean_reversion']=.5+(row['id']//2)*.2
    dataset.write_text(json.dumps({k:v for k,v in payload.items() if k!='results'},indent=2)[:-2]
        +',\n  "results": [\n'+',\n'.join(json.dumps(r) for r in payload['results'])+'\n]\n}\n')
    write_yaml(dataset.with_suffix('.recipe.yaml'),{'shape':{'parameter_count':4,
        'paths_per_parameter':2,'row_order':'parameter-major, then path-major'}})
    c,v,meta=load_reference(ref,'cir')
    c[0]=[.5,.04,.1,.03,1.0]
    save_reference(ref,c,v,meta)
    with pytest.raises(ValueError,match='policy'):prepare(root,recipe)
    recipe['data']['training_rows']='exclude_reference_parameters'
    write_yaml(path,recipe)
    data=prepare(root,recipe)
    assert data.manifest['checks']['excluded_reference_rows']==2
    assert data.manifest['checks']['train_parameter_groups']==3
    assert selection(data).manifest()['counts']=={'train':6,'validation':32,'test':32}
    np.testing.assert_array_equal(data.parameter_groups[:6],[0,0,1,1,2,2])
    assert set(data.parameter_groups[:6]).isdisjoint(data.parameter_groups[6:])
    np.testing.assert_array_equal(data.conditions[6:],np.repeat(c,16,axis=0))
    np.testing.assert_array_equal(data.observations[6:,0],v[:,:16].ravel())
    run_recipe(path,root)
    import torch
    checkpoint=torch.load(collection_path(root,recipe['output']['model'],'gan'),weights_only=False)
    assert checkpoint['transform']['condition_mean'][0]==pytest.approx(.9)
    assert prepare(root,recipe).fingerprint==data.fingerprint


def test_public_gan_catalogue_rejects_grouped_source(tiny):
    from learning.gan.catalogue import create_leaf
    root,path,recipe,ref=tiny
    dataset=collection_path(root,recipe['dataset']['path'],'datasets')
    write_yaml(dataset.with_suffix('.recipe.yaml'), {'shape':{'parameter_count':4,
        'paths_per_parameter':2,'row_order':'parameter-major, then path-major'}})
    with pytest.raises(ValueError, match='one path per parameter'):
        create_leaf(root,dataset)


def test_missing_stored_reference_has_no_deleted_dataset_dependency(tiny):
    from learning.gan.references import create_reference
    root,path,recipe,ref=tiny
    ref.unlink()
    with pytest.raises(FileNotFoundError,match='reference archive'):
        create_reference(root,recipe)
