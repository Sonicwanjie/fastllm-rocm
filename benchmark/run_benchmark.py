"""
run_benchmark.py - End-to-end decode benchmark suite for fastllm-rocm

NOTE: As of 2026-05-27, only the following models can actually run:
- Gemma 4 E2B BF16 (safetensors): Loads, but inference crashes (PLE not complete)
- Gemma 4 E2B GGUF: Fails to load (unsupported weight types)
- Qwen3.6-27B GGUF: Architecture not supported (hybrid SSM+attention)

The benchmark scripts are ready for when models become functional.
This script will test whichever models are loadable and report results.

Usage:
    python benchmark/run_benchmark.py --model gemma4-e2b
    python benchmark/run_benchmark.py --list
"""

import subprocess, time, sys, os, threading, queue, json, argparse
from pathlib import Path
from datetime import datetime

EXE_NINJA = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\main.exe"
EXE_RELEASE = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\Release\main.exe"
ROCMBIN = r"C:\ROCm\bin"
WORKDIR = r"C:\Users\q\.openclaw\workspace\fastllm-rocm"
RESULT_DIR = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\benchmark\results"

MODELS = {
    "gemma4-e2b": {
        "name": "Gemma 4 E2B BF16 (safetensors)",
        "path": r"C:\Users\q\.openclaw\workspace\fastllm-rocm\models\gemma-4-e2b-it",
        "type": "hf",
        "dtype": "bfloat16",
        "atype": "float16",
        "size_gb": 9.55,
        "exe": "release",
        "status": "loads_but_crashes_on_inference",
    },
    "gemma4-e2b-gguf": {
        "name": "Gemma 4 E2B Q4_K_M (GGUF)",
        "path": r"C:\Users\q\.lmstudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_M.gguf",
        "type": "gguf",
        "dtype": None,
        "atype": None,
        "size_gb": 3.19,
        "exe": "ninja",
        "status": "load_fails_unsupported_weight",
    },
}

TESTS = [
    {"id": "math", "prompt": "What is 2+3? Answer with just the number.", "expected": ["5"]},
    {"id": "factual", "prompt": "What is the capital of France?", "expected": ["Paris"]},
    {"id": "repeat", "prompt": "Repeat exactly: HELLO WORLD", "expected": ["HELLO WORLD"]},
]


def get_env():
    env = os.environ.copy()
    env["PATH"] = ROCMBIN + ";" + env.get("PATH", "")
    return env


def get_exe(model_key):
    m = MODELS[model_key]
    if m["exe"] == "ninja" and os.path.exists(EXE_NINJA):
        return EXE_NINJA
    return EXE_RELEASE


def build_cmd(model_key):
    m = MODELS[model_key]
    exe = get_exe(model_key)
    cmd = [exe, "-p", m["path"], "-t", "4"]
    if m["dtype"]:
        cmd += ["--dtype", m["dtype"]]
    if m["atype"]:
        cmd += ["--atype", m["atype"]]
    return cmd


def run_model_test(model_key, prompt, max_wait=120):
    """Start model, send prompt, collect response."""
    m = MODELS[model_key]
    cmd = build_cmd(model_key)
    env = get_env()

    print(f"  CMD: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, cwd=WORKDIR,
    )

    q = queue.Queue()
    def reader():
        for raw in proc.stdout:
            try:
                line = raw.decode("utf-8", errors="replace").rstrip()
            except:
                line = repr(raw)
            q.put(line)

    threading.Thread(target=reader, daemon=True).start()

    def drain():
        lines = []
        while True:
            try:
                lines.append(q.get_nowait())
            except queue.Empty:
                break
        return lines

    # Wait for load
    print("  Waiting for model load...", flush=True)
    load_start = time.perf_counter()
    all_lines = []
    while time.perf_counter() - load_start < 600:
        time.sleep(1)
        lines = drain()
        all_lines.extend(lines)

        if any("\u7528\u6237" in l for l in lines[-5:]):
            load_time = time.perf_counter() - load_start
            print(f"  Loaded in {load_time:.1f}s", flush=True)
            break

        for l in lines:
            if "Error" in l and "chatTemplate" not in l:
                print(f"  ERROR during load: {l[:120]}", flush=True)
                proc.kill()
                return {"error": "load_error", "lines": all_lines, "detail": l}

        if proc.poll() is not None:
            print(f"  Process exited during load: {proc.returncode}", flush=True)
            return {"error": "load_crash", "lines": all_lines}
    else:
        proc.kill()
        return {"error": "load_timeout", "lines": all_lines}

    # Send prompt
    print(f"  Sending prompt: {prompt[:60]}...", flush=True)
    proc.stdin.write((prompt + "\n").encode("utf-8"))
    proc.stdin.flush()

    t_start = time.perf_counter()
    response = []
    token_times = []
    first_token = None
    last_t = t_start

    while time.perf_counter() - t_start < max_wait:
        time.sleep(0.05)
        lines = drain()
        all_lines.extend(lines)
        for l in lines:
            now = time.perf_counter()
            if "\u7528\u6237" in l:
                # Response done
                elapsed = now - t_start
                ttft = (first_token - t_start) * 1000 if first_token else 0
                proc.stdin.write(b"stop\n")
                proc.stdin.flush()
                try:
                    proc.wait(timeout=5)
                except:
                    proc.kill()
                text = "".join(response)
                return {
                    "text": text,
                    "tokens": len(token_times),
                    "elapsed_ms": elapsed * 1000,
                    "ttft_ms": ttft,
                    "tokens_per_sec": len(token_times) / elapsed if elapsed > 0 else 0,
                    "token_times": token_times,
                    "all_lines": all_lines,
                }

            if "Error" in l:
                print(f"  ERROR during inference: {l[:120]}", flush=True)
                proc.kill()
                return {"error": "inference_error", "detail": l, "lines": all_lines}

            content = l.strip()
            if content and not content.startswith("Load") and not content.startswith("Loading"):
                if first_token is None:
                    first_token = now
                itl = (now - last_t) * 1000
                token_times.append(itl)
                response.append(content)
                last_t = now

    proc.kill()
    return {"error": "timeout", "lines": all_lines}


def run_correctness(model_key):
    m = MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"CORRECTNESS TEST: {m['name']}")
    print(f"Status: {m['status']}")
    print(f"{'='*70}")

    if not os.path.exists(m["path"]):
        print(f"  SKIP: Model not found")
        return None

    result = run_model_test(model_key, "What is 2+3? Answer with just the number.", max_wait=60)

    if "error" in result:
        print(f"\n  RESULT: {result['error']}")
        if "detail" in result:
            print(f"  Detail: {result['detail'][:200]}")
        result["model"] = model_key
        result["model_name"] = m["name"]
        result["timestamp"] = datetime.now().isoformat()
        return result

    print(f"\n  Response: {result['text'][:200]}")
    print(f"  Tokens: {result['tokens']} | Time: {result['elapsed_ms']:.0f}ms | Speed: {result['tokens_per_sec']:.1f} tok/s")

    result["model"] = model_key
    result["model_name"] = m["name"]
    result["timestamp"] = datetime.now().isoformat()
    return result


def main():
    parser = argparse.ArgumentParser(description="fastllm-rocm benchmark")
    parser.add_argument("--model", default="gemma4-e2b", help="Model key")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        for k, v in MODELS.items():
            exists = os.path.exists(v["path"])
            print(f"  {k:<25} {v['name']:<40} exists={exists} status={v['status']}")
        return

    os.makedirs(RESULT_DIR, exist_ok=True)

    if args.correctness or True:  # Default to correctness test
        r = run_correctness(args.model)
        if r:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(RESULT_DIR, f"{args.model}_test_{ts}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)
            print(f"\n  Results saved: {path}")


if __name__ == "__main__":
    main()
