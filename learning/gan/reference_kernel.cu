// Fixed-condition references reuse each model's sample dynamics and preparation.
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#if MODEL == 0
#include "model/equity/markovian/heston/dynamics_impl.cuh"
namespace m = ai_factory::workbench::model::equity::heston;
using Policy=m::DynamicsPolicy;
constexpr int columns=6;
__device__ auto parameters(const float* p) { return m::ModelParameters{1,0,0,p[4],p[0],p[2],p[1],p[3]}; }
#elif MODEL == 1
#include "model/equity/markovian/bates/dynamics_impl.cuh"
namespace m = ai_factory::workbench::model::equity::bates;
using Policy=m::DynamicsPolicy;
constexpr int columns=9;
__device__ auto parameters(const float* p) { return m::ModelParameters{1,0,0,p[4],p[0],p[2],p[1],p[3],p[5],p[6],p[7]}; }
#elif MODEL == 2
#include "model/equity/rough/rough_heston/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/sample_device_preparation.cuh"
namespace m = ai_factory::workbench::model::equity::rough_heston;
using Policy=m::DynamicsPolicy<7>;
constexpr int columns=7;
__device__ auto parameters(const float* p) { return m::ModelParameters{1,0,0,p[0],p[1],p[2],p[3],p[4],p[5]}; }
#elif MODEL == 3
#include "model/fixed_income/cir/dynamics_impl.cuh"
namespace m = ai_factory::workbench::model::fixed_income::cir;
using Policy=m::DynamicsPolicy;
constexpr int columns=5;
__device__ auto parameters(const float* p) { return m::ModelParameters{{p[0],p[1],p[2]},p[3]}; }
#endif
#if MODEL < 0 || MODEL > 3
#error Unsupported GAN reference model
#endif
namespace rng=ai_factory::workbench::philox;
void checked(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__global__ void prepare(const float* c, Policy::PreparedDynamics* output) {
 if(threadIdx.x) return;
 const auto* p=c+blockIdx.x*columns;
#if MODEL == 2
 output[blockIdx.x]=m::SampleDevicePreparer<7>::prepare(parameters(p), {2.f,1.f/504.f});
#elif MODEL == 3
 output[blockIdx.x]=Policy::prepare_dynamics(parameters(p),p[columns-1]);
#else
 output[blockIdx.x]=Policy::prepare_dynamics(parameters(p),1.f/504.f);
#endif
}
__global__ void generate(const float* c, const Policy::PreparedDynamics* prepared, float* out, int paths, unsigned long long seed) {
 const int row=blockIdx.y, path=blockIdx.x*blockDim.x+threadIdx.x;
 if(path>=paths) return;
 auto state=Policy::initial_state(prepared[row]);
 Policy::RandomContext random(rng::make_key(seed+row),path);
#if MODEL == 3
 Policy::advance(prepared[row],1,random,state);
 out[row*paths+path]=state;
#else
 Policy::advance(prepared[row],__float2uint_rn(c[row*columns+columns-1]*504.f),random,state);
 out[row*paths+path]=Policy::spot(state);
#endif
}
int main(int argc,char** argv) {
 if(argc!=6) return 2;
 int rows=std::stoi(argv[3]),paths=std::stoi(argv[4]);
 if(rows<=0 || paths<=0) return 3;
 std::vector<float> c(rows*columns),out(rows*paths);
 std::ifstream input(argv[1],std::ios::binary);
 input.read(reinterpret_cast<char*>(c.data()),c.size()*sizeof(float));
 if(!input) return 3;
 float *dc,*dout; Policy::PreparedDynamics* dp;
 checked(cudaMalloc(&dc,c.size()*sizeof(float)));
 checked(cudaMalloc(&dout,out.size()*sizeof(float)));
 checked(cudaMalloc(&dp,rows*sizeof(*dp)));
 checked(cudaMemcpy(dc,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
 prepare<<<rows,32>>>(dc,dp); checked(cudaGetLastError());
 generate<<<dim3((paths+255)/256,rows),256>>>(dc,dp,dout,paths,std::stoull(argv[5]));
 checked(cudaGetLastError());
 checked(cudaMemcpy(out.data(),dout,out.size()*sizeof(float),cudaMemcpyDeviceToHost));
 std::ofstream output(argv[2],std::ios::binary);
 output.write(reinterpret_cast<const char*>(out.data()),out.size()*sizeof(float));
 checked(cudaFree(dc));checked(cudaFree(dp));checked(cudaFree(dout));
 return output ? 0:4;
}
