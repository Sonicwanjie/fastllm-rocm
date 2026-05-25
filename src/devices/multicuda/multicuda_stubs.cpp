// Stubs for multi-CUDA / NCCL functions not available in ROCm single-GPU build
// These are referenced by multicudadevice.cpp and executor.cpp but not needed for single GPU

#include "fastllm.h"
#include <hip/hip_runtime.h>
#include <vector>
#include <map>
#include <string>
#include <cstring>

using DivisionScheme = std::map<int, std::vector<std::pair<int, int>>>;

extern "C" {

bool FastllmInitNccl(const std::vector<int>&) { return false; }
void FastllmNcclBroadcast(void*, int, int, int, int) {}
void FastllmNcclAllReduce(void*, void*, int, int, int) {}
void FastllmCudaSyncDevice(int) {}

std::vector<int> FastllmMultiCudaGetSplitPoints(std::vector<int>&, std::map<int, int>&, int, int) {
    return {};
}

void FastllmGetMulticudaDeviceAndRatio(std::vector<int>& devices, std::map<int, int>& ratios, bool) {
    devices.clear();
    ratios.clear();
}

bool SplitMultiCudaWeight(fastllm::Data&, fastllm::Data&, std::vector<int>&, DivisionScheme, int) {
    return false;
}

bool SplitMultiCudaWeight1D(fastllm::Data&, std::vector<int>&, DivisionScheme) {
    return false;
}

bool PlaceMultiCudaWeightOnDevice(fastllm::Data&, std::vector<int>&, int) {
    return false;
}

void CopyToMultiDevices(fastllm::Data&, std::vector<int>, bool) {}
void PrepareMultiCudaReplicatedData(fastllm::Data&, std::vector<int>, bool) {}

void PrepareMultiCudaShardedData(fastllm::Data&, std::vector<int>,
    const std::vector<int>&, int, DivisionScheme) {}

DivisionScheme BuildMultiCudaRowSplitScheme(fastllm::Data&, std::vector<int>&, std::map<int, int>&) {
    return {};
}

void FastllmMultiCudaSetDevice(std::vector<int>) {}
void FastllmMultiCudaSetDeviceRatio(std::map<int, int>&) {}

bool FastllmMultiCudaHalfMatMul(const fastllm::Data&, fastllm::Data&, const fastllm::Data&, fastllm::Data&, int, int, int) {
    return false;
}

bool FastllmMultiCudaMatMul(const fastllm::Data&, fastllm::Data&, const fastllm::Data&, fastllm::Data&, int, int, int) {
    return false;
}

} // extern "C"

// FastllmCudaMemcpy2DDeviceToDeviceAuto is a multi-GPU wrapper not defined in .hip files
// Declared in extern "C" block in fastllm-cuda.cuh
extern "C" {
void FastllmCudaMemcpy2DDeviceToDeviceAuto(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, int srcDevice, int dstDevice) {
    // Single GPU: just do a regular device-to-device copy
    hipMemcpy2D(dst, dpitch, src, spitch, width, height, hipMemcpyDeviceToDevice);
}
} // extern "C"

// Additional stubs for functions referenced by multicudadevice.cpp
namespace fastllm {
    void EnsureReplicatedMultiCudaTensor(Data&, const std::vector<int>&, bool) {}
    bool RunMultiCudaRowLinear(Data&, Data&, Data&, Data&) { return false; }
}
