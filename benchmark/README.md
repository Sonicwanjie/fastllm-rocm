# Benchmark Suite

End-to-end decode benchmark suite for fastllm-rocm on AMD Radeon 8060S (gfx1151).

## Scripts

### `run_benchmark.py` - Main Benchmark Runner
Runs correctness, decode throughput, and kernel probe tests on any available model.

```bash
# Run all benchmarks on Qwen3.6 (should work correctly)
python benchmark/run_benchmark.py --model qwen36

# Run only decode throughput test
python benchmark/run_benchmark.py --model qwen36 --decode

# Run only correctness tests
python benchmark/run_benchmark.py --model qwen36 --correctness

# Run all models and compare
python benchmark/run_benchmark.py --model all --compare

# List available models
python benchmark/run_benchmark.py --list
```

### `probe_decode_timing.py` - FP16/BF16 Decode Timing Probes
High-resolution per-token timing analysis for decode path.

```bash
# Probe Qwen3.6 decode timing
python benchmark/probe_decode_timing.py --model qwen36

# Probe Gemma 4 GGUF
python benchmark/probe_decode_timing.py --model gemma4-e2b-gguf

# Compare all models
python benchmark/probe_decode_timing.py --compare-all
```

Outputs:
- Per-token latency distribution (avg, p50, p90, p95, p99)
- Time-to-first-token (TTFT / prefill latency)
- Steady-state decode throughput (tokens/sec)
- Comparison table across models

### `bench_gemma4_decode.py` - Gemma 4 E2B Specific Benchmark
Gemma 4 specific tests including PLE, KV-shared layers, and partial rotary validation.

```bash
# Benchmark GGUF format
python benchmark/bench_gemma4_decode.py --format gguf

# Benchmark safetensors BF16
python benchmark/bench_gemma4_decode.py --format safetensors --dtype bfloat16

# Benchmark safetensors INT4
python benchmark/bench_gemma4_decode.py --format safetensors --dtype int4

# All formats
python benchmark/bench_gemma4_decode.py --all
```

### `benchmark_timer.h` - Internal GPU Timer (C++)

Add `#define FASTLLM_ENABLE_BENCH_TIMER` and include this header in the
linear/attention kernel files to get per-operation GPU timing via `hipEvent`.

Reports kernel-level timing at program exit:
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FASTLLM KERNEL BENCH TIMER REPORT                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Operation                                Calls   Total(ms)   Avg(ms)  ... ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Linear_FP16_decode                          50      21.00     0.420  ... ║
║ Attention_decode                            50       7.50     0.150  ... ║
║ RMSNorm                                    100       2.00     0.020  ... ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Results

All results are saved to `benchmark/results/` as JSON files with timestamps.

## Current Baseline (as of 2026-05-26)

| Model | Format | Correctness | Decode Speed | Target |
|-------|--------|-------------|-------------|--------|
| Qwen3.6-27B | GGUF Q4_K_M | TBD | TBD | >40 tok/s |
| Gemma 4 E2B | GGUF Q4_K_M | TBD (PLE) | TBD | >40 tok/s |
| Gemma 4 E2B | BF16 safetensors | FAIL (乱码) | ~20-23 tok/s | >40 tok/s |
| Gemma 4 E2B | INT4 safetensors | TBD | TBD | >80 tok/s |

## Kernel Comparison

The "old" kernels use WMMA 16x16 (low utilization for M=1 decode).
The "new" tilelang kernels use MFMA with fused dequant/attention.

Timing probes measure the external effect of these kernel paths.
For internal kernel timing, use `benchmark_timer.h` with a rebuild.
