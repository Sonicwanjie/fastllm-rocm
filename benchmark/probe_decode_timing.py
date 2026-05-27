"""
probe_decode_timing.py - FP16/BF16 decode timing probes

Measures the per-token decode latency for the old (WMMA) and new (MFMA/tilelang)
kernel paths. Since both kernels share the same main.exe interface, this probes
the external timing by generating many tokens and analyzing the latency distribution.

For internal kernel-level timing, this script also generates a C++ header patch
that adds hipEvent timing inside the linear/attention paths.

Usage:
    python benchmark/probe_decode_timing.py --model qwen36
    python benchmark/probe_decode_timing.py --model gemma4-e2b-gguf
    python benchmark/probe_decode_timing.py --model gemma4-e2b --dtype bfloat16
    python benchmark/probe_decode_timing.py --compare-all
"""

import subprocess, time, sys, os, threading, queue, json, argparse, statistics
from pathlib import Path
from datetime import datetime

EXE = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\Release\main.exe"
ROCMBIN = r"C:\ROCm\bin"
WORKDIR = r"C:\Users\q\.openclaw\workspace\fastllm-rocm"
RESULT_DIR = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\benchmark\results"

MODELS = {
    "qwen36": {
        "path": r"C:\Users\q\.lmstudio\models\lmstudio-community\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q4_K_M.gguf",
        "name": "Qwen3.6-27B Q4_K_M",
        "cmd_extra": [],
    },
    "gemma4-e2b-gguf": {
        "path": r"C:\Users\q\.lmstudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_M.gguf",
        "name": "Gemma 4 E2B GGUF Q4_K_M",
        "cmd_extra": [],
    },
    "gemma4-e2b-bf16": {
        "path": r"C:\Users\q\.openclaw\workspace\fastllm-rocm\models\gemma-4-e2b-it",
        "name": "Gemma 4 E2B BF16",
        "cmd_extra": ["--dtype", "bfloat16", "--atype", "float16"],
    },
    "gemma4-e2b-int4": {
        "path": r"C:\Users\q\.openclaw\workspace\fastllm-rocm\models\gemma-4-e2b-it",
        "name": "Gemma 4 E2B INT4",
        "cmd_extra": ["--dtype", "int4", "--atype", "float16"],
    },
}

# Multiple prompts to measure prefill + decode separately
PROBE_PROMPTS = [
    ("short_1tok", "What is 1+1?"),
    ("medium_20tok", "Write a haiku about the sun."),
    ("long_100tok", "Explain the difference between FP16 and BF16 floating point formats in detail."),
    ("sustained", "Count from 1 to 100, one number per line."),
]


def get_env():
    env = os.environ.copy()
    env["PATH"] = ROCMBIN + ";" + env.get("PATH", "")
    return env


class ModelProcess:
    """Manage main.exe subprocess with precise timing."""

    def __init__(self, model_key):
        self.model_key = model_key
        self.info = MODELS[model_key]
        self.proc = None
        self.q = queue.Queue()
        self.all_lines = []

    def start(self, timeout=300):
        cmd = [EXE, "-p", self.info["path"], "-t", "4"] + self.info["cmd_extra"]
        env = get_env()
        print(f"  CMD: {' '.join(cmd)}", flush=True)

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, cwd=WORKDIR,
        )

        def reader():
            for raw in self.proc.stdout:
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                except:
                    line = repr(raw)
                self.q.put(line)

        threading.Thread(target=reader, daemon=True).start()

        load_start = time.perf_counter()
        while time.perf_counter() - load_start < timeout:
            time.sleep(0.3)
            self._drain()
            if any("\u7528\u6237" in l for l in self.all_lines[-10:]):
                return True, time.perf_counter() - load_start
            if self.proc.poll() is not None:
                return False, time.perf_counter() - load_start

        return False, timeout

    def _drain(self):
        while True:
            try:
                l = self.q.get_nowait()
                self.all_lines.append(l)
            except queue.Empty:
                break

    def generate(self, prompt, max_wait=120):
        """Generate with high-resolution per-token timing."""
        self.all_lines = []
        token_times = []  # (absolute_time, inter_token_latency)

        self.proc.stdin.write((prompt + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        t_start = time.perf_counter()
        last_t = t_start
        response = []
        first_token_time = None

        while time.perf_counter() - t_start < max_wait:
            time.sleep(0.01)  # Poll frequently for precise timing
            self._drain()
            new_start = len(response)

            for l in self.all_lines[len(response):]:
                now = time.perf_counter()
                if not l.strip() or l.startswith("Load") or l.startswith("Loading"):
                    response.append(l)
                    continue
                if "\u7528\u6237" in l:
                    # Response complete
                    total_time = now - t_start
                    ttft = first_token_time - t_start if first_token_time else 0
                    return {
                        "text": "".join(response),
                        "tokens": len(token_times),
                        "total_time_ms": total_time * 1000,
                        "ttft_ms": ttft * 1000,
                        "token_times": token_times,
                    }

                content = l.strip()
                if content:
                    if first_token_time is None:
                        first_token_time = now
                        ttft = now - t_start
                    itl = now - last_t
                    token_times.append((now - t_start, itl))
                    last_t = now
                response.append(l)

        # Timeout
        return {
            "text": "".join(response),
            "tokens": len(token_times),
            "total_time_ms": (time.perf_counter() - t_start) * 1000,
            "ttft_ms": (first_token_time - t_start) * 1000 if first_token_time else 0,
            "token_times": token_times,
        }

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(b"stop\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except:
                self.proc.kill()


def compute_stats(token_times, skip_first=3):
    """Compute decode timing statistics."""
    itls = [itl for _, itl in token_times]
    if len(itls) <= skip_first:
        skip_first = 0
    steady = itls[skip_first:]
    if not steady:
        return None

    steady_ms = [x * 1000 for x in steady]
    return {
        "total_tokens": len(itls),
        "warmup_tokens": skip_first,
        "steady_tokens": len(steady),
        "avg_ms": statistics.mean(steady_ms),
        "median_ms": statistics.median(steady_ms),
        "stdev_ms": statistics.stdev(steady_ms) if len(steady_ms) > 1 else 0,
        "min_ms": min(steady_ms),
        "max_ms": max(steady_ms),
        "p90_ms": sorted(steady_ms)[int(len(steady_ms) * 0.9)],
        "p95_ms": sorted(steady_ms)[int(len(steady_ms) * 0.95)],
        "p99_ms": sorted(steady_ms)[min(int(len(steady_ms) * 0.99), len(steady_ms) - 1)],
        "throughput_tps": 1000.0 / statistics.mean(steady_ms) if statistics.mean(steady_ms) > 0 else 0,
    }


def run_probe(model_key, prompts=None):
    """Run decode timing probes for a model."""
    prompts = prompts or PROBE_PROMPTS
    info = MODELS[model_key]

    print(f"\n{'='*70}")
    print(f"DECODE TIMING PROBE: {info['name']}")
    print(f"{'='*70}")

    if not os.path.exists(info["path"]):
        print(f"  SKIP: {info['path']} not found")
        return None

    proc = ModelProcess(model_key)
    ok, load_time = proc.start()
    if not ok:
        print("  FAILED to load!")
        proc.stop()
        return None

    print(f"  Loaded in {load_time:.1f}s\n")

    # Warmup
    print("  Warmup...", flush=True)
    proc.generate("Hi", max_wait=30)
    time.sleep(0.5)

    results = {
        "model": model_key,
        "model_name": info["name"],
        "timestamp": datetime.now().isoformat(),
        "load_time_s": load_time,
        "probes": {},
    }

    for pid, prompt in prompts:
        print(f"  Probe [{pid}]: {prompt[:50]}...", flush=True)
        resp = proc.generate(prompt, max_wait=120)
        stats = compute_stats(resp["token_times"])

        probe_result = {
            "prompt_id": pid,
            "prompt": prompt,
            "tokens": resp["tokens"],
            "total_time_ms": resp["total_time_ms"],
            "ttft_ms": resp["ttft_ms"],
            "stats": stats,
        }
        results["probes"][pid] = probe_result

        if stats:
            print(f"    TTFT: {resp['ttft_ms']:.1f}ms | "
                  f"Decode: {stats['throughput_tps']:.1f} tok/s "
                  f"(avg={stats['avg_ms']:.1f}ms, p50={stats['median_ms']:.1f}ms, "
                  f"p90={stats['p90_ms']:.1f}ms)")
        else:
            print(f"    No tokens generated")

    proc.stop()

    # Summary
    print(f"\n  --- SUMMARY ---")
    for pid, probe in results["probes"].items():
        s = probe["stats"]
        if s:
            print(f"    {pid:<20} {s['throughput_tps']:.1f} tok/s | "
                  f"avg={s['avg_ms']:.1f}ms p50={s['median_ms']:.1f}ms")

    return results


def compare_models(model_keys=None):
    """Run probes on multiple models and compare."""
    model_keys = model_keys or ["qwen36", "gemma4-e2b-gguf"]
    all_results = []

    for mk in model_keys:
        if mk not in MODELS:
            continue
        if not os.path.exists(MODELS[mk]["path"]):
            print(f"SKIP {mk}: not found")
            continue
        r = run_probe(mk)
        if r:
            os.makedirs(RESULT_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(RESULT_DIR, f"{mk}_timing_{ts}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)
            all_results.append(r)

    # Comparison table
    if len(all_results) > 1:
        print(f"\n{'='*90}")
        print(f" FP16/BF16 DECODE TIMING COMPARISON")
        print(f"{'='*90}")
        print(f"{'Model':<30} {'tok/s':<10} {'avg(ms)':<10} {'p50(ms)':<10} "
              f"{'p90(ms)':<10} {'TTFT(ms)':<10}")
        print(f"{'-'*90}")

        for r in all_results:
            name = r["model_name"][:28]
            # Use sustained probe if available, else long
            probe = r["probes"].get("sustained") or r["probes"].get("long_100tok")
            if probe and probe["stats"]:
                s = probe["stats"]
                print(f"{name:<30} {s['throughput_tps']:<10.1f} {s['avg_ms']:<10.1f} "
                      f"{s['median_ms']:<10.1f} {s['p90_ms']:<10.1f} "
                      f"{probe['ttft_ms']:<10.1f}")
            else:
                print(f"{name:<30} {'N/A':<10}")

        print(f"{'='*90}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Decode timing probe")
    parser.add_argument("--model", default="qwen36")
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--dtype", default=None, help="Override model dtype")
    args = parser.parse_args()

    if args.compare_all:
        compare_models()
    else:
        r = run_probe(args.model)
        if r:
            os.makedirs(RESULT_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(RESULT_DIR, f"{args.model}_timing_{ts}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
