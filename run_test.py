import subprocess, time, sys, os

model_path = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\models\gemma-4-e2b-it"
exe = r"C:\Users\q\.openclaw\workspace\fastllm-rocm\build-rocm-msvc2\Release\main.exe"
outf = open(r"C:\Users\q\.openclaw\workspace\fastllm-rocm\test_output.txt", "w", encoding="utf-8")

env = os.environ.copy()
env["PATH"] = r"C:\ROCm\bin;" + env.get("PATH", "")

proc = subprocess.Popen(
    [exe, "--path", model_path, "--dtype", "bfloat16"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    env=env, cwd=r"C:\Users\q\.openclaw\workspace\fastllm-rocm"
)

import threading

lines = []
def reader():
    for raw in proc.stdout:
        try:
            line = raw.decode("utf-8", errors="replace").rstrip()
        except:
            line = repr(raw)
        lines.append(line)
        outf.write(line + "\n")
        outf.flush()

t = threading.Thread(target=reader, daemon=True)
t.start()

print("Waiting for load...", flush=True)
deadline = time.time() + 600
while time.time() < deadline:
    time.sleep(1)
    if any("\u7528\u6237" in l or "user" in l.lower() for l in lines[-5:]):
        print("LOADED!", flush=True)
        break
    if proc.poll() is not None:
        print(f"Exit during load: {proc.returncode}", flush=True)
        break
    if len(lines) % 100 == 0 and len(lines) > 0:
        print(f"  ... {len(lines)} lines, last: {lines[-1][:50] if lines else 'none'}", flush=True)

if proc.poll() is None:
    time.sleep(2)
    print("Sending prompt...", flush=True)
    proc.stdin.write(b"Hello\n")
    proc.stdin.flush()
    
    start = time.time()
    while time.time() - start < 120:
        time.sleep(3)
        if proc.poll() is not None:
            print(f"Process exited: {proc.returncode}", flush=True)
            break
        if len(lines) > 0:
            last = lines[-1]
            print(f"  [{int(time.time()-start)}s] last output: {last[:100]}", flush=True)

    if proc.poll() is None:
        proc.stdin.write(b"stop\n")
        proc.stdin.flush()
        time.sleep(3)
        if proc.poll() is None:
            proc.kill()

time.sleep(2)
outf.close()

print("\n=== OUTPUT (last 30 lines) ===", flush=True)
with open(r"C:\Users\q\.openclaw\workspace\fastllm-rocm\test_output.txt", "r", encoding="utf-8") as f:
    all_lines = f.readlines()
    for l in all_lines[-30:]:
        print(l.rstrip(), flush=True)
