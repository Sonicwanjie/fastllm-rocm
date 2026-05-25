// sampling.cuh - Redirect to flashinfer sampling for ROCm HIP builds
#pragma once

#ifdef USE_ROCM
// For ROCm, include the real flashinfer sampling from third_party
// The include path -I third_party/flashinfer should make this work
// but since this stub is found first via -I include/devices/cuda,
// we need to explicitly include from the flashinfer directory.
// Use a relative path from this file's location to find flashinfer
#include "../../../third_party/flashinfer/sampling.cuh"
#else
// Original CUDA sampling header placeholder
// Add implementations as needed when porting sampling features
#endif
