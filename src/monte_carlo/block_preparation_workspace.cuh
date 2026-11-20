// Optional per-block storage for deterministic path preparation.
#pragma once
#include "cuda/check_cuda.cuh"
#include <cstddef>
#include <limits>
#include <stdexcept>
namespace ai_factory::workbench::monte_carlo {
struct BlockPreparationWorkspace {
    unsigned char* data = nullptr;
    std::size_t stride = 0U;
    __device__ unsigned char* block(std::size_t index) const {
        return data == nullptr ? nullptr : data + index * stride;
    }
};
class BlockPreparationStorage {
    BlockPreparationWorkspace view_{};
public:
    BlockPreparationStorage(std::size_t blocks, std::size_t stride) {
        if (stride == 0U || blocks == 0U) return;
        if (blocks > std::numeric_limits<std::size_t>::max() / stride)
            throw std::overflow_error("Block preparation workspace overflow.");
        view_.stride = stride;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.data), blocks * stride),
                   "Allocate block preparation workspace");
    }
    BlockPreparationStorage(const BlockPreparationStorage&) = delete;
    BlockPreparationStorage& operator=(const BlockPreparationStorage&) = delete;
    ~BlockPreparationStorage() { if (view_.data != nullptr) cudaFree(view_.data); }
    BlockPreparationWorkspace view() const { return view_; }
};
} // namespace ai_factory::workbench::monte_carlo
