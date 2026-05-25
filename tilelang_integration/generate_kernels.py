"""
TileLang Kernel Generator for fastllm Integration

Generates CUDA/HIP kernel source files from tilelang DSL,
to be compiled and linked into fastllm's device layer.

AOT Pipeline:
  1. tilelang DSL (Python) -> compile -> kernel source (.cu/.hip)
  2. Generated source -> hipcc/cl.exe compile -> .obj
  3. .obj linked into fastllm.dll/fastllm.lib

Usage:
    python generate_kernels.py --target hip --output-dir generated
    python generate_kernels.py --target hip --kernel dequant_gemv_int4
    python generate_kernels.py --target hip --compile --compiler "clang++"
    python generate_kernels.py --manifest  # just print registry
"""

import os
import sys
import argparse
import json
from pathlib import Path

# ============================================================================
# Kernel Registry (without tilelang import for manifest mode)
# ============================================================================

KERNEL_REGISTRY = {
    "dequant_gemv_int4": {
        "factory_args": {"M": 1, "N": 4096, "K": 4096},
        "desc": "INT4 dequantize GEMV (decode, M=1)",
        "configs": [
            {"N": 4096, "K": 4096},
            {"N": 14336, "K": 4096},
            {"N": 4096, "K": 14336},
        ],
    },
    "dequant_gemm_int4_batch": {
        "factory_args": {"M": 32, "N": 4096, "K": 4096},
        "desc": "INT4 dequantize GEMM (prefill, M>1)",
        "configs": [
            {"M": 32, "N": 4096, "K": 4096},
        ],
    },
    "flash_decode": {
        "factory_args": {"batch": 1, "heads": 32, "groups": 8, "seqlen_kv": 4096, "dim": 128},
        "desc": "Flash Decoding for GQA decode",
        "configs": [
            {"heads": 32, "groups": 8, "dim": 128},
            {"heads": 32, "groups": 32, "dim": 256},  # Gemma 4 sliding
            {"heads": 32, "groups": 4, "dim": 512},   # Gemma 4 global
        ],
    },
    "fused_moe": {
        "factory_args": {"num_experts": 64, "top_k": 8, "hidden_size": 4096, "inter_size": 14336},
        "desc": "Fused MoE kernel",
        "configs": [
            {"num_experts": 64, "top_k": 8, "hidden_size": 4096, "inter_size": 14336},
        ],
    },
    "mla_decode": {
        "factory_args": {"batch": 1, "heads": 128, "seqlen_kv": 4096, "q_dim": 192, "kv_dim": 512},
        "desc": "DeepSeek MLA decode kernel",
        "configs": [
            {"heads": 128, "seqlen_kv": 4096, "q_dim": 192, "kv_dim": 512},
        ],
    },
}


def try_import_tilelang():
    """Attempt to import tilelang. Returns None if unavailable."""
    try:
        import tilelang
        import tilelang.language as T
        return tilelang, T
    except ImportError as e:
        print(f"Warning: tilelang not available ({e})")
        print("Running in manifest-only mode. Install tilelang for code generation.")
        return None, None


# ============================================================================
# Kernel factory functions (require tilelang)
# ============================================================================

def make_dequant_gemv_int4(M=1, N=4096, K=4096, **kwargs):
    tilelang, T = try_import_tilelang()
    if tilelang is None:
        return None
    from tilelang.quantize import _tir_packed_to_unsigned_convert

    storage_dtype = T.int8
    num_bits = 4
    num_elems_per_byte = 8 // num_bits
    block_N = 64
    block_K = 64

    A_shape = (K,)
    B_shape = (N, K // num_elems_per_byte)
    C_shape = (N,)

    @T.prim_func
    def dequant_gemv(
        A: T.Tensor(A_shape, "float16"),
        B: T.Tensor(B_shape, storage_dtype),
        C: T.Tensor(C_shape, "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=128) as (bx,):
            A_shared = T.alloc_shared([block_K], "float16")
            B_shared = T.alloc_shared([block_N, block_K // num_elems_per_byte], storage_dtype)
            C_local = T.alloc_fragment([block_N], "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K // num_elems_per_byte], B_shared)
                for j in T.Parallel(block_N):
                    acc = T.alloc_fragment([1], "float32")
                    acc[0] = T.float32(0)
                    for ki in T.Serial(block_K):
                        q_val = T.cast(A_shared[ki], "float32")
                        w_bits = B_shared[j, (ki) // num_elems_per_byte]
                        w_val = _tir_packed_to_unsigned_convert("int", 8)(
                            num_bits, w_bits, ki % num_elems_per_byte, dtype="float16")
                        acc[0] += q_val * T.cast(w_val, "float32")
                    C_local[j] += acc[0]
            T.copy(C_local, C[bx * block_N])

    return dequant_gemv

def make_dequant_gemm_int4_batch(M=32, N=4096, K=4096, **kwargs):
    tilelang, T = try_import_tilelang()
    if tilelang is None:
        return None
    from tilelang.quantize import _tir_packed_to_unsigned_convert

    num_bits = 4
    num_elems_per_byte = 8 // num_bits
    block_M, block_N, block_K = 64, 64, 64

    @T.prim_func
    def dequant_gemm(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K // num_elems_per_byte), T.int8),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_K], "float16")
            B_shared = T.alloc_shared([block_N, block_K // num_elems_per_byte], T.int8)
            B_dequant = T.alloc_fragment([block_N, block_K], "float16")
            C_local = T.alloc_fragment([block_M, block_N], "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K // num_elems_per_byte], B_shared)
                for i, j in T.Parallel(block_N, block_K):
                    B_dequant[i, j] = _tir_packed_to_unsigned_convert("int", 8)(
                        num_bits, B_shared[i, j // num_elems_per_byte],
                        j % num_elems_per_byte, dtype="float16")
                T.gemm(B_dequant, A_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return dequant_gemm


def make_flash_decode(batch=1, heads=32, groups=8, seqlen_kv=4096, dim=128, **kwargs):
    tilelang, T = try_import_tilelang()
    if tilelang is None:
        return None

    group_size = heads // groups

    @T.prim_func
    def flash_decode(
        Q: T.Tensor((batch, heads, dim), "float16"),
        K: T.Tensor((batch, groups, seqlen_kv, dim), "float16"),
        V: T.Tensor((batch, groups, seqlen_kv, dim), "float16"),
        O: T.Tensor((batch, heads, dim), "float16"),
    ):
        with T.Kernel(heads, threads=32) as (head_idx,):
            kv_head = head_idx // group_size
            Q_local = T.alloc_fragment([dim], "float32")
            acc_o = T.alloc_fragment([dim], "float32")
            scores_max = T.alloc_fragment([1], "float32")
            scores_max_prev = T.alloc_fragment([1], "float32")
            logsum = T.alloc_fragment([1], "float32")
            T.copy(Q[0, head_idx, :], Q_local)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity("float32"))
            for ko in T.Pipelined(T.ceildiv(seqlen_kv, 1), num_stages=1):
                acc_s = T.alloc_fragment([1], "float32")
                K_local = T.alloc_fragment([dim], "float16")
                V_local = T.alloc_fragment([dim], "float16")
                T.copy(K[0, kv_head, ko, :], K_local)
                acc_s[0] = 0
                for d in T.Serial(dim):
                    acc_s[0] += Q_local[d] * T.cast(K_local[d], "float32")
                acc_s[0] *= (1.0 / (dim ** 0.5))
                T.copy(scores_max, scores_max_prev)
                scores_max[0] = T.max(scores_max[0], acc_s[0])
                for d in T.Parallel(dim):
                    acc_o[d] *= T.exp2(scores_max_prev[0] - scores_max[0])
                acc_s[0] = T.exp2(acc_s[0] - scores_max[0])
                T.copy(V[0, kv_head, ko, :], V_local)
                for d in T.Serial(dim):
                    acc_o[d] += acc_s[0] * T.cast(V_local[d], "float32")
                logsum[0] = logsum[0] * T.exp2(scores_max_prev[0] - scores_max[0])
                logsum[0] += acc_s[0]
            for d in T.Parallel(dim):
                O[0, head_idx, d] = T.cast(acc_o[d] / (logsum[0] + 1e-10), "float16")

    return flash_decode


def make_fused_moe(num_experts=64, top_k=8, hidden_size=4096, inter_size=14336, **kwargs):
    tilelang, T = try_import_tilelang()
    if tilelang is None:
        return None

    @T.prim_func
    def fused_moe(
        X: T.Tensor((1, hidden_size), "float16"),
        W_gate: T.Tensor((num_experts, inter_size, hidden_size), "float16"),
        W_up: T.Tensor((num_experts, inter_size, hidden_size), "float16"),
        W_down: T.Tensor((num_experts, hidden_size, inter_size), "float16"),
        Router: T.Tensor((hidden_size, num_experts), "float16"),
        O: T.Tensor((1, hidden_size), "float16"),
    ):
        # This is a placeholder — actual fused MoE in tilelang requires
        # custom routing logic that goes beyond pure TIR
        pass

    return fused_moe


def make_mla_decode(batch=1, heads=128, seqlen_kv=4096, q_dim=192, kv_dim=512, **kwargs):
    tilelang, T = try_import_tilelang()
    if tilelang is None:
        return None

    @T.prim_func
    def mla_decode(
        Q: T.Tensor((batch, heads, q_dim), "float16"),
        K: T.Tensor((batch, seqlen_kv, kv_dim), "float16"),
        V: T.Tensor((batch, seqlen_kv, kv_dim), "float16"),
        O: T.Tensor((batch, heads, q_dim), "float16"),
    ):
        with T.Kernel(heads, threads=32) as (head_idx,):
            Q_local = T.alloc_fragment([q_dim], "float32")
            acc_o = T.alloc_fragment([q_dim], "float32")
            scores_max = T.alloc_fragment([1], "float32")
            logsum = T.alloc_fragment([1], "float32")
            T.copy(Q[0, head_idx, :], Q_local)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity("float32"))
            for ko in T.Serial(seqlen_kv):
                K_proj = T.alloc_fragment([q_dim], "float16")
                V_row = T.alloc_fragment([q_dim], "float16")
                for di in T.Serial(q_dim):
                    K_proj[di] = K[0, ko, di]
                acc_s = T.alloc_fragment([1], "float32")
                acc_s[0] = T.float32(0)
                for di in T.Serial(q_dim):
                    acc_s[0] += Q_local[di] * T.cast(K_proj[di], "float32")
                acc_s[0] *= T.float32(1.0 / (q_dim ** 0.5))
                scores_max_prev = scores_max[0]
                scores_max[0] = T.max(scores_max[0], acc_s[0])
                correction = T.exp2(scores_max_prev - scores_max[0])
                for di in T.Parallel(q_dim):
                    acc_o[di] *= correction
                acc_s[0] = T.exp2(acc_s[0] - scores_max[0])
                for di in T.Serial(q_dim):
                    V_row[di] = V[0, ko, di]
                for di in T.Parallel(q_dim):
                    acc_o[di] += acc_s[0] * T.cast(V_row[di], "float32")
                logsum[0] = logsum[0] * correction
                logsum[0] += acc_s[0]
            inv_sum = T.float32(1.0) / (logsum[0] + T.float32(1e-10))
            for di in T.Parallel(q_dim):
                O[0, head_idx, di] = T.cast(acc_o[di] * inv_sum, "float16")

    return mla_decode

# ============================================================================
# Factory dispatch
# ============================================================================

FACTORY_MAP = {
    "dequant_gemv_int4": make_dequant_gemv_int4,
    "dequant_gemm_int4_batch": make_dequant_gemm_int4_batch,
    "flash_decode": make_flash_decode,
    "fused_moe": make_fused_moe,
    "mla_decode": make_mla_decode,
}


# ============================================================================
# AOT compilation helpers
# ============================================================================

def generate_kernel_source(name: str, target: str = "hip", mcpu: str = "gfx1151", **factory_kwargs) -> str:
    """Generate HIP source for a named kernel (AOT: source only, no JIT link)."""
    factory = FACTORY_MAP.get(name)
    if factory is None:
        raise ValueError(f"Unknown kernel: {name}. Available: {list(FACTORY_MAP.keys())}")

    prim_func = factory(**factory_kwargs)
    if prim_func is None:
        raise RuntimeError(f"tilelang not available, cannot generate source for {name}")

    import tvm
    from tilelang.engine.lower import lower
    compiled_target = tvm.target.Target({"kind": target, "mcpu": mcpu}) if mcpu else target
    with compiled_target:
        artifact = lower(prim_func, target=compiled_target)
    if not artifact.kernel_source:
        raise RuntimeError(f"No kernel source generated for {name}")
    return artifact.kernel_source

def compile_to_object(source_path: Path, output_path: Path, compiler: str = "clang++",
                      target: str = "hip") -> bool:
    """Compile generated source to object file."""
    import subprocess
    cmd = [
        compiler,
        "-c", str(source_path),
        "-o", str(output_path),
        "-O3",
        "-std=c++17",
    ]
    if target == "hip":
        cmd.extend(["-x", "hip", "--offload-arch=gfx1151"])
    elif target == "cuda":
        cmd.extend(["-x", "cu", "--gpu-architecture=sm_80"])

    print(f"  Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  COMPILE ERROR: {result.stderr}")
        return False
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate tilelang kernels for fastllm")
    parser.add_argument("--target", choices=["cuda", "hip"], default="hip",
                        help="Target backend")
    parser.add_argument("--output-dir", default="generated",
                        help="Output directory for generated sources")
    parser.add_argument("--kernels", nargs="*", default=None,
                        help="Specific kernels to generate (default: all)")
    parser.add_argument("--compile", action="store_true",
                        help="Also compile to .obj files")
    parser.add_argument("--compiler", default="clang++",
                        help="Compiler for --compile step")
    parser.add_argument("--manifest", action="store_true",
                        help="Print kernel manifest and exit")
    args = parser.parse_args()

    if args.manifest:
        print(json.dumps(KERNEL_REGISTRY, indent=2))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kernels = args.kernels or list(KERNEL_REGISTRY.keys())
    manifest = {}
    generated_count = 0
    failed_count = 0

    for name in kernels:
        entry = KERNEL_REGISTRY[name]
        print(f"Generating {name}: {entry['desc']}")

        try:
            source = generate_kernel_source(name, target=args.target, **entry["factory_args"])
        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  (tilelang may not be available; kernel will use AOT bridge fallback)")
            failed_count += 1
            continue

        ext = "cu" if args.target == "cuda" else "hip"
        filename = f"tl_{name}.{ext}"
        filepath = output_dir / filename
        filepath.write_text(source, encoding="utf-8")
        print(f"  -> {filepath} ({len(source)} bytes)")

        obj_path = None
        if args.compile:
            obj_filename = f"tl_{name}.o" if args.target == "hip" else f"tl_{name}.obj"
            obj_path = output_dir / obj_filename
            if compile_to_object(filepath, obj_path, args.compiler, args.target):
                print(f"  -> {obj_path}")
            else:
                obj_path = None

        manifest[name] = {
            "file": filename,
            "obj": str(obj_path) if obj_path else None,
            "desc": entry["desc"],
            "target": args.target,
            "configs": entry.get("configs", []),
        }
        generated_count += 1

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nGenerated {generated_count}/{len(kernels)} kernels "
          f"({failed_count} failed, using bridge fallback).")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
