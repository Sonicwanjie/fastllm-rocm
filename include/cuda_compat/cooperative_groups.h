#pragma once
// Cooperative groups stub for HIP builds
#include <hip/hip_runtime.h>

namespace cooperative_groups {
    struct thread_block {
        __device__ void sync() const { __syncthreads(); }
    };
    inline __device__ thread_block this_thread_block() { return thread_block(); }
    
    template <typename T>
    struct thread_group {
        __device__ void sync() const {}
    };
}
