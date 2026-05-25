<!--
### fastllm_hip_types_minimal.hpp
Minimal type forward declarations for HIP device compilation.
Replaces fastllm.h when compiling kernels with hipcc (avoids MSVC STL issues on Windows).
Only used when __HIPCC__ is defined.
-->

#ifndef FASTLLM_HIP_TYPES_MINIMAL_H
#define FASTLLM_HIP_TYPES_MINIMAL_H

#include <cstdint>
#include <stddef.h>

namespace fastllm {

enum DataType {
    FLOAT32 = 0, BFLOAT16 = 1, INT16 = 2, INT8 = 3, INT4 = 4, INT2 = 5, BIT = 6, FLOAT16 = 7,
    INT4_NOZERO = 8, INT4_GROUP = 9, FP8_E4M3 = 10, INT2_GROUP = 11, BASE3_GROUP = 12,
    INT32 = 13, NVFP4 = 14, INT32PARAM = 100, FP8_E4M3_BLOCK_128 = 1000,
    AWQ_4BIT_128 = 1001, INT4_PERCHANNEL = 1002, FP8_E4M3_PERCHANNEL = 1003,
    INT4_GROUP128 = 1004, INT8_PERCHANNEL = 1005, NVFP4_BLOCK_16 = 1006,
    NVFP4_BLOCK_16_E8M0 = 1007,
    INF_INT8_PERCHANNEL = 2000, INF_INT8_GROUP128 = 2001,
    DATA_GGUF_FORMAT = 9999, DATA_GGUF_FORMAT_END = 19999,
    DATA_AUTO_NONE = 99999, DATA_AUTO_LINEAR, DATA_AUTO_EMBEDDING, DATA_AUTO_CONV
};

enum DataDevice { CPU = 0, CUDA = 1 };

// NOTE: This is a STRIPPED-DOWN view of fastllm::Data for kernel use.
// Host code must use the real fastllm.h definition.
// The fields below are accessed by wrapper functions that are compiled
// in the SAME .hip translation unit. For function parameters passed by
// reference (const fastllm::Data&), the layout must MATCH the real definition.
// Therefore we declare `Data` as opaque and only use it through extern functions.
class Data;

} // namespace fastllm

#endif // FASTLLM_HIP_TYPES_MINIMAL_H