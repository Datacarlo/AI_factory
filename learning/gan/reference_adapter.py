"""Fixed-condition driver around existing sample launchers (no duplicated dynamics)."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import numpy as np
import torch
from .models import MODELS


def source_for(model):
    spec = MODELS[model]
    namespace = ('equity' if spec.path.startswith('equity/') else 'fixed_income')
    fft = '/rough/' in spec.path and model != 'quadratic_rough_heston'
    lift = model == 'quadratic_rough_heston'
    fields = []
    for i, name in enumerate(spec.conditions[:-1]):
        target = name
        if namespace == 'fixed_income':
            if model == 'g2' and name.startswith('initial_state_'):
                target = 'initial_state.state_'+name[-1]
            elif name != 'initial_state':
                target = 'process.'+name
        fields.append(f'parameter.{target}=p[{i}];')
    if 'spot' in spec.fixed: fields.append('parameter.spot=1.0f;')
    secondary = model in ('heston_3_2', 'sabr', 'schobel_zhu', 'g2')
    outputs = 'first, second' if secondary else 'first'
    arguments = 'dp,replicas,paths_per_replica,days,0,allocated,' + ('' if fft else '256,') + 'blocks,seed+(static_cast<unsigned long long>(row)<<32),'+outputs
    call = f'm::launch_{model}_terminal_samples_cuda'+('<7>' if lift else '')+f'({arguments});'
    prepare = 'std::vector<m::ModelParameters> repeated(replicas,parameter); checked(cudaMemcpy(dp,repeated.data(),replicas*sizeof(parameter),cudaMemcpyHostToDevice));'
    extra = ''
    storage = 'm::ModelParameters'
    if lift:
        storage = 'm::PreparedDynamics<7>'
        extra = '''__global__ void prepare_reference(m::ModelParameters parameter, m::PreparedDynamics<7>* out, int replicas) {
          if(threadIdx.x==0) { auto fitted=m::SampleDevicePreparer<7>::prepare(parameter,{2.f,1.f/504.f});
            for(int i=0;i<replicas;++i) out[i]=fitted; }
        }'''
        prepare = 'prepare_reference<<<1,32>>>(parameter,dp,replicas); checked(cudaGetLastError());'
    return f'''// Generated adapter: source sample launcher, explicit conditions and stable row seed.
#include "model/{spec.path}/sample.cu"
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cmath>
namespace m=ai_factory::workbench::model::{namespace}::{model};
void checked(cudaError_t e) {{ if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }}
{extra}
int main(int argc,char** argv) {{
 if(argc!=6) return 2;
 int rows=std::stoi(argv[3]),paths=std::stoi(argv[4]);
 if(rows<=0 || paths<=0) return 3;
 auto seed=std::stoull(argv[5]);
 constexpr int columns={len(spec.conditions)}, dimensions={len(spec.observables)};
 // Fixed replica width preserves prefix identity when the requested draw count changes.
 constexpr int paths_per_replica=64;
 const int replicas=(paths+paths_per_replica-1)/paths_per_replica, allocated=replicas*paths_per_replica;
 std::vector<float> conditions(rows*columns), a(allocated), b(allocated), output(rows*paths*dimensions);
 std::ifstream input(argv[1],std::ios::binary);
 input.read(reinterpret_cast<char*>(conditions.data()),conditions.size()*sizeof(float));
 if(!input) return 4;
 {storage}* dp; float *first,*second;
 checked(cudaMalloc(&dp,replicas*sizeof(*dp))); checked(cudaMalloc(&first,allocated*sizeof(float)));
 checked(cudaMalloc(&second,allocated*sizeof(float)));
 for(int row=0;row<rows;++row) {{
   auto* p=conditions.data()+row*columns;
   m::ModelParameters parameter{{}};
   {''.join(fields)}
   auto days=static_cast<unsigned int>(std::lround(p[columns-1]*252.0));
   {prepare}
   const std::size_t blocks=replicas;
   {call}
   checked(cudaGetLastError());
   checked(cudaMemcpy(a.data(),first,allocated*sizeof(float),cudaMemcpyDeviceToHost));
   if constexpr(dimensions==2) checked(cudaMemcpy(b.data(),second,allocated*sizeof(float),cudaMemcpyDeviceToHost));
   for(int i=0;i<paths;++i) {{ output[(row*paths+i)*dimensions]=a[i];
     if constexpr(dimensions==2) output[(row*paths+i)*dimensions+1]=b[i]; }}
 }}
 std::ofstream out(argv[2],std::ios::binary);
 out.write(reinterpret_cast<const char*>(output.data()),output.size()*sizeof(float));
 checked(cudaFree(dp)); checked(cudaFree(first)); checked(cudaFree(second));
 return out ? 0:5;
}}
'''


def mathdx_flags(root, compiler, architecture):
    configured = os.environ.get('AI_FACTORY_MATHDX_ROOT')
    major = re.search(r'release (\d+)\.', subprocess.check_output([compiler,'--version'],text=True)).group(1)
    candidates = [Path(configured)] if configured else []
    for directory in (root.parent/'.toolchains', Path.home()/'opt'):
        candidates.extend(directory.glob(f'*mathdx*cuda{major}/nvidia/mathdx/*'))
    for path in candidates:
        if (path/'include/cufftdx.hpp').is_file():
            return ['-I'+str(path/'include'),'-I'+str(path/'external/cutlass/include'),
                    '-DAI_FACTORY_HAS_CUFFTDX=1','-DAI_FACTORY_CUFFTDX_ARCHITECTURE='+str(int(architecture)*10)]
    raise FileNotFoundError('Set AI_FACTORY_MATHDX_ROOT to the MathDx distribution matching nvcc')


def simulate_native(root, model, conditions, *, paths, seed, work, nvcc, architecture):
    spec=MODELS[model]
    compiler=shutil.which(nvcc)
    if not compiler: raise FileNotFoundError('nvcc required for native references')
    architecture=str(architecture or ''.join(map(str,torch.cuda.get_device_capability())))
    if not architecture.isdigit(): raise ValueError('Expected numeric CUDA architecture')
    work=Path(work); work.mkdir(parents=True,exist_ok=True)
    source=work/'adapter_kernel.cu'; source.write_text(source_for(model))
    binary=work/'reference_simulator'
    flags=mathdx_flags(root,compiler,architecture) if '/rough/' in spec.path and model!='quadratic_rough_heston' else []
    command=[compiler,'-std=c++20','-O3','-arch=sm_'+architecture,'-I'+str(root/'src'),*flags,str(source),str(root/'src/cuda/cuda_kernel_diagnostics.cpp'),'-o',str(binary)]
    # No persistent binary cache: compile against the current source when creating a reference.
    with (work/'build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    input_path,output_path=work/'conditions.bin',work/'values.bin'
    conditions.tofile(input_path)
    subprocess.run([str(binary),str(input_path),str(output_path),str(len(conditions)),str(paths),str(seed)],check=True)
    shape=(len(conditions),paths) if len(spec.observables)==1 else (len(conditions),paths,len(spec.observables))
    values=np.fromfile(output_path,dtype=np.float32).reshape(shape)
    if not np.isfinite(values).all() or (spec.nonnegative and np.any(values<0)):
        raise ValueError('Native reference contains invalid observations')
    return values
