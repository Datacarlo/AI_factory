// Byte strides retain the existing per-row workspace layout when batching rows.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <type_traits>
namespace ai_factory::workbench::volterra {
template<class T>
__device__ __forceinline__ T* row_workspace_pointer(T* pointer, std::size_t stride) {
    using Byte = std::conditional_t<std::is_const_v<T>, const unsigned char, unsigned char>;
    return reinterpret_cast<T*>(reinterpret_cast<Byte*>(pointer) + blockIdx.y * stride);
}
}
