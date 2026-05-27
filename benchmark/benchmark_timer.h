/**
 * benchmark_timer.h - GPU timing probes for FP16/BF16 decode path
 *
 * Usage: Include this header in cudadevice.cpp or the linear kernel files.
 * The FASTLLM_BENCH_TIMER macro wraps kernel calls with hipEvent timing.
 *
 * To enable: define FASTLLM_ENABLE_BENCH_TIMER before including.
 * Results print to stdout at program exit.
 *
 * Example output:
 *   [BENCH] Linear_FP16 decode  n=1 m=1536 k=6144  avg=0.42ms  calls=50
 *   [BENCH] Attention_decode    seqlen=128 heads=8  avg=0.15ms  calls=50
 */

#ifndef FASTLLM_BENCH_TIMER_H
#define FASTLLM_BENCH_TIMER_H

#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <mutex>

#ifdef FASTLLM_ENABLE_BENCH_TIMER

namespace fastllm {
namespace bench {

struct TimerResult {
    std::string name;
    int calls = 0;
    double total_ms = 0.0;
    double min_ms = 1e9;
    double max_ms = 0.0;
    double last_ms = 0.0;
};

class TimerRegistry {
public:
    static TimerRegistry& Instance() {
        static TimerRegistry reg;
        return reg;
    }

    void* CreateEvent() {
        void* ev = nullptr;
        // hipEventCreate with timing enabled (NOT disable timing)
        hipEventCreate((hipEvent_t*)&ev);
        return ev;
    }

    void Record(void* ev, void* stream) {
        hipEventRecord((hipEvent_t)ev, (hipStream_t)stream);
    }

    float Elapsed(void* start, void* end) {
        float ms = 0;
        hipEventSynchronize((hipEvent_t)end);
        hipEventElapsedTime(&ms, (hipEvent_t)start, (hipEvent_t)end);
        return ms;
    }

    void RecordTiming(const std::string& name, float ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& r = results_[name];
        r.name = name;
        r.calls++;
        r.total_ms += ms;
        r.last_ms = ms;
        if (ms < r.min_ms) r.min_ms = ms;
        if (ms > r.max_ms) r.max_ms = ms;
    }

    void PrintReport() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (results_.empty()) return;

        printf("\n");
        printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        printf("║                    FASTLLM KERNEL BENCH TIMER REPORT                       ║\n");
        printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        printf("║ %-40s %8s %10s %10s %10s %10s ║\n",
               "Operation", "Calls", "Total(ms)", "Avg(ms)", "Min(ms)", "Max(ms)");
        printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");

        for (auto& [name, r] : results_) {
            double avg = r.calls > 0 ? r.total_ms / r.calls : 0;
            printf("║ %-40s %8d %10.2f %10.3f %10.3f %10.3f ║\n",
                   name.c_str(), r.calls, r.total_ms, avg, r.min_ms, r.max_ms);
        }

        printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        printf("\n");
    }

    ~TimerRegistry() {
        PrintReport();
    }

private:
    std::mutex mutex_;
    std::map<std::string, TimerResult> results_;
};

// Scoped timer: creates events, records start/end, computes elapsed
class ScopedTimer {
public:
    ScopedTimer(const std::string& name, void* stream = nullptr)
        : name_(name), stream_(stream) {
        auto& reg = TimerRegistry::Instance();
        start_ = reg.CreateEvent();
        end_ = reg.CreateEvent();
        reg.Record(start_, stream_);
    }

    ~ScopedTimer() {
        auto& reg = TimerRegistry::Instance();
        reg.Record(end_, stream_);
        float ms = reg.Elapsed(start_, end_);
        reg.RecordTiming(name_, ms);
        hipEventDestroy((hipEvent_t)start_);
        hipEventDestroy((hipEvent_t)end_);
    }

private:
    std::string name_;
    void* stream_;
    void* start_;
    void* end_;
};

} // namespace bench
} // namespace fastllm

// Macros for easy use
#define BENCH_TIMER_BEGIN(name, stream) \
    fastllm::bench::ScopedTimer _bench_timer_##name(#name, stream)

#define BENCH_TIMER_RECORD(name, extra_info) \
    ; // extra_info is for logging only

#else // FASTLLM_ENABLE_BENCH_TIMER not defined

// No-op macros when timer is disabled
#define BENCH_TIMER_BEGIN(name, stream)
#define BENCH_TIMER_RECORD(name, extra_info)

#endif // FASTLLM_ENABLE_BENCH_TIMER

#endif // FASTLLM_BENCH_TIMER_H
