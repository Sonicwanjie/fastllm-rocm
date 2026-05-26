// Windows stubs for Linux NUMA headers
// Included by numasdevice.cpp and numas.cpp on Windows builds

#ifndef FASTLLM_NUMAS_WIN_H
#define FASTLLM_NUMAS_WIN_H

#ifdef _WIN32

// Stub Linux NUMA types and constants so code compiles
// Actual NUMA operations fall back to standard Windows APIs

#define MPOL_BIND 0

typedef struct {
    unsigned long* maskp;
    unsigned long size;
} bitmask;

static inline int numa_available() { return -1; }
static inline int numa_run_on_node(int node) { return -1; }
static inline int numa_num_configured_cpus() { return 0; }
static inline bitmask* numa_allocate_cpumask() { return nullptr; }
static inline void numa_free_cpumask(bitmask* bm) {}
static inline int numa_node_to_cpus(int node, bitmask* bm) { return -1; }
static inline int numa_bitmask_isbitset(bitmask* bm, unsigned int n) { return 0; }
static inline bitmask* numa_allocate_nodemask() { return nullptr; }
static inline void numa_free_nodemask(bitmask* bm) {}
static inline void numa_bitmask_setbit(bitmask* bm, unsigned int n) {}
static inline int set_mempolicy(int mode, unsigned long* nodemask, unsigned long maxnode) { return -1; }
static inline void* numa_alloc_onnode(size_t size, int node) { return _aligned_malloc(size, 64); }
static inline void numa_free(void* ptr, size_t size) { _aligned_free(ptr); }

// CPU affinity stubs
typedef struct { unsigned long __bits[16]; } cpu_set_t;
static inline void CPU_ZERO(cpu_set_t* s) { memset(s, 0, sizeof(*s)); }
static inline void CPU_SET(int cpu, cpu_set_t* s) { if (cpu >= 0 && cpu < 128) s->__bits[cpu/64] |= (1UL << (cpu%64)); }
static inline int sched_setaffinity(int pid, size_t size, cpu_set_t* s) { return 0; }

#endif // _WIN32
#endif // FASTLLM_NUMAS_WIN_H
