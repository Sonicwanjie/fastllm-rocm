//
// Windows implementation of NUMAS functions
// Replaces Linux numas.cpp
//

#include "devices/numas/numas_win.h"
#include "devices/cpu/alivethreadpool.h"
#include "utils.h"

#include <cstdlib>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fastllm {

MachineNumaInfo::MachineNumaInfo() {
#ifdef _WIN32
    ULONG highestNode = 0;
    if (!GetNumaHighestNodeNumber(&highestNode)) highestNode = 0;
    numaCnt = (int)highestNode + 1;
    if (numaCnt < 1) numaCnt = 1;

    cpuIds.resize(numaCnt);
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    int totalCpus = (int)sysInfo.dwNumberOfProcessors;

    for (int cpu = 0; cpu < totalCpus && cpu < 256; cpu++) {
        PROCESSOR_NUMBER procNum = {};
        procNum.Number = (UCHAR)cpu;
        USHORT node = 0;
        if (GetNumaProcessorNodeEx(&procNum, &node)) {
            if ((int)node < numaCnt) cpuIds[node].push_back(cpu);
        } else {
            cpuIds[0].push_back(cpu);
        }
    }
#else
    numaCnt = 1;
    cpuIds.resize(1);
#endif
}

NumaConfig::NumaConfig() {}

NumaConfig::NumaConfig(int numThreads, AliveThreadPool *pool, MachineNumaInfo *info) {
    this->numaCnt = info->numaCnt;
    this->threads = numThreads;
    this->numaToCpuDict.resize(this->numaCnt);
    this->threadIdToNumaDict.resize(this->threads, 0);
    int per = this->threads / this->numaCnt;
    if (per < 1) per = 1;
    int threadIdx = 0;
    for (int i = 0; i < this->numaCnt; i++) {
        int cpusForNode = std::min(per, (int)info->cpuIds[i].size());
        for (int j = 0; j < cpusForNode && threadIdx < this->threads; j++) {
            this->threadIdToNumaDict[threadIdx] = i;
            this->numaToCpuDict[i].push_back(std::make_pair(threadIdx, info->cpuIds[i][j]));
            threadIdx++;
        }
    }
}

void bind_to_cpu(int cpu_id) {
#ifdef _WIN32
    if (cpu_id >= 0 && cpu_id < 64)
        SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu_id);
#endif
}

void bind_to_numa_node(int node_id) {}
void set_numa_mempolicy(int node_id) {}

void* allocate_aligned(size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, 64);
#else
    void* p = nullptr; posix_memalign(&p, 64, size); return p;
#endif
}

void free_aligned(void* ptr, size_t size) {
    if (!ptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void* allocate_aligned_numa(size_t size, int node) {
#ifdef _WIN32
    return VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    return allocate_aligned(size);
#endif
}

void free_aligned_numa(void* ptr, size_t size) {
    if (!ptr) return;
#ifdef _WIN32
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    free_aligned(ptr, size);
#endif
}

void* allocate_pinned_numa(size_t size, int node) {
    void* ptr = allocate_aligned_numa(size, node);
#ifdef USE_CUDA
    if (ptr) FastllmCudaHostRegister(ptr, size);
#endif
    return ptr;
}

void free_pinned_numa(void* ptr, size_t size) {
#ifdef USE_CUDA
    FastllmCudaHostUnregister(ptr);
#endif
    free_aligned_numa(ptr, size);
}

}
