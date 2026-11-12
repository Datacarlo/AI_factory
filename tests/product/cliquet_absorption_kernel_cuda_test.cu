// Payoff regression: absorption, local bounds and shared-path finite differences.
#include "product/cliquet/pricing_policy.cuh"
#include "cuda/check_cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Policy = ai_factory::workbench::product::CliquetPathPolicy;
using ai_factory::workbench::check_cuda;
struct Case { float spots[5]; float participation, floor, cap, global_floor, global_cap; };
struct SpotState { struct State {}; };
struct Result { float price; unsigned positive_parity; };

__global__ void evaluate(const Case* cases, Result* results, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const Case c = cases[i];
    Policy::Handler h{c.participation, c.floor, c.cap};
    h.on_initial_value(c.spots[0]);
    float legacy = 0.0f;
    bool positive = true;
    for (unsigned j = 1; j < 5; ++j) {
        h.on_observation(j - 1, c.spots[j]);
        positive = positive && c.spots[j - 1] > 0 && c.spots[j] > 0;
        legacy += fminf(c.cap, fmaxf(c.floor,
            c.participation * (c.spots[j] / c.spots[j - 1] - 1.0f)));
    }
    Policy::PreparedProduct p{c.participation,c.floor,c.cap,c.global_floor,c.global_cap,1.0f};
    results[i] = {Policy::finalize<SpotState>(p, {}, h),
        unsigned(!positive || __float_as_uint(legacy) == __float_as_uint(h.accumulated_return))};
}

double reference(const Case& c) {
    double sum = 0;
    for (unsigned j = 1; j < 5; ++j) {
        // Independent contract definition: charge the first loss, then zero return.
        double r = c.spots[j - 1] == 0 && c.spots[j] == 0
            ? 0 : double(c.spots[j]) / c.spots[j - 1] - 1;
        sum += std::clamp(double(c.participation) * r, double(c.floor), double(c.cap));
    }
    return 1 + std::clamp(sum, double(c.global_floor), double(c.global_cap));
}
}
int main() {
    try {
        std::vector<Case> cases{
            {{1,0,0,0,0},1,-.0625f,.125f,-1,1},
            {{1,.5f,0,0,0},1,-.0625f,.125f,-1,1},
            {{1,1,1,1,1},1,-.0625f,.125f,-1,1},
            // Bounds still apply to a zero return (even a positive local floor).
            {{1,0,0,0,0},1,.03125f,.125f,-1,1},
            {{1,0,0,0,0},1,-.0625f,.125f,-.03125f,1},
        };
        const auto deterministic = cases.size();
        // Fixed innovations, different spot bumps: absorption time changes.
        for (float shift : {-.03125f, 0.f, .03125f}) {
            Case c{{1 + shift, 0, 0, 0, 0},1,-.5f,.5f,-1,1};
            c.spots[1] = std::max(0.f, .03125f + shift);
            c.spots[2] = c.spots[1] > 0 ? std::max(0.f, shift) : 0;
            cases.push_back(c);
        }
        for (float floor : {-.078125f, -.0625f, -.046875f})
            cases.push_back({{1,0,0,0,0},1,floor,.125f,-1,1});
        for (unsigned i = 0; i < 4096; ++i) {
            float x = 1.f + float(i) / 4096;
            cases.push_back({{x,x*.9375f,x*1.0625f,x*.875f,x},.875f,-.0625f,.125f,-1,1});
        }
        Case* dc=nullptr; Result* dr=nullptr;
        check_cuda(cudaMalloc(&dc,cases.size()*sizeof(Case)),"cases");
        check_cuda(cudaMalloc(&dr,cases.size()*sizeof(Result)),"results");
        check_cuda(cudaMemcpy(dc,cases.data(),cases.size()*sizeof(Case),cudaMemcpyHostToDevice),"upload");
        evaluate<<<(cases.size()+127)/128,128>>>(dc,dr,cases.size());
        check_cuda(cudaGetLastError(),"evaluate");
        std::vector<Result> results(cases.size());
        check_cuda(cudaMemcpy(results.data(),dr,results.size()*sizeof(Result),cudaMemcpyDeviceToHost),"download");
        check_cuda(cudaFree(dc),"free cases");check_cuda(cudaFree(dr),"free results");
        for (unsigned i=0;i<cases.size();++i) {
            if (!std::isfinite(results[i].price) || std::abs(results[i].price-reference(cases[i]))>3e-7)
                throw std::runtime_error("Contract mismatch at case " + std::to_string(i));
            if (!results[i].positive_parity) throw std::runtime_error("Positive-path bitwise regression");
        }
        if (results[0].price != .9375f) throw std::runtime_error("Repeated loss after absorption");
        for (unsigned base : {unsigned(deterministic),unsigned(deterministic+3)}) {
            double h=base==deterministic?.03125:.015625;
            double g=(results[base+2].price-results[base].price)/(2*h);
            double expected=(reference(cases[base+2])-reference(cases[base]))/(2*h);
            if (std::abs(g-expected)>1e-5) throw std::runtime_error("Shared-path gradient mismatch");
        }
        std::cout << cases.size() << " contract cases passed; 4096 positive paths bitwise unchanged; two gradient stencils passed\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
