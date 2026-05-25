# generate_flash_attn.py
# Generate optimized flash attention HIP kernels from tilelang DSL
# for integration into fastllm
#
# Usage:
#   python generate_flash_attn.py --output-dir ../generated
#
# This generates HIP source files that can be compiled and linked
# directly into fastllm, replacing the hand-written bridge kernel.

import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
import argparse
import json
from pathlib import Path


def make_flash_attn_gqa_256(batch, heads_q, heads_kv, seq_q, seq_kv, dim=256, is_causal=True):
    """Generate flash attention kernel for Gemma-4 (head_dim=256, GQA)."""
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e) for exp2 optimization
    group_size = heads_q // heads_kv

    q_shape = [batch, heads_q, seq_q, dim]
    kv_shape = [batch, heads_kv, seq_kv, dim]

    @autotune(configs=[
        dict(block_M=128, block_N=128, num_stages=2, threads=256),
        dict(block_M=64, block_N=128, num_stages=2, threads=256),
        dict(block_M=64, block_N=64, num_stages=3, threads=256),
    ], warmup=5, rep=5)
    @tilelang.jit(
        out_idx=[3],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def flash_attn(
        Q: T.Tensor(q_shape, "float16"),
        K: T.Tensor(kv_shape, "float16"),
        V: T.Tensor(kv_shape, "float16"),
        Output: T.Tensor(q_shape, "float16"),
        block_M=64, block_N=64, num_stages=2, threads=128,
    ):
        with T.Kernel(
            T.ceildiv(seq_q, block_M), heads_q, batch, threads=threads
        ) as (bx, by, bz):
            kv_head = by // group_size

            Q_shared = T.alloc_shared([block_M, dim], "float16")
            K_shared = T.alloc_shared([block_N, dim], "float16")
            V_shared = T.alloc_shared([block_N, dim], "float16")
            O_shared = T.alloc_shared([block_M, dim], "float16")

            acc_s = T.alloc_fragment([block_M, block_N], "float32")
            acc_s_cast = T.alloc_fragment([block_M, block_N], "float16")
            acc_o = T.alloc_fragment([block_M, dim], "float32")

            scores_max = T.alloc_fragment([block_M], "float32")
            scores_max_prev = T.alloc_fragment([block_M], "float32")
            scores_scale = T.alloc_fragment([block_M], "float32")
            scores_sum = T.alloc_fragment([block_M], "float32")
            logsum = T.alloc_fragment([block_M], "float32")

            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity("float32"))

            past_len = seq_kv - seq_q

            loop_range = (
                T.min(
                    T.ceildiv(seq_kv, block_N),
                    T.ceildiv((bx + 1) * block_M + past_len, block_N)
                ) if is_causal else T.ceildiv(seq_kv, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, kv_head, k * block_N:(k + 1) * block_N, :], K_shared)

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i + past_len
                        k_idx = k * block_N + j
                        acc_s[i, j] = T.if_then_else(
                            q_idx >= k_idx, 0, -T.infinity(acc_s.dtype)
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * block_N + j >= seq_kv,
                            -T.infinity(acc_s.dtype), 0
                        )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity("float32"))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                T.reduce_sum(acc_s, scores_sum, dim=1)

                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, kv_head, k * block_N:(k + 1) * block_N, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]

            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

    return flash_attn


def main():
    parser = argparse.ArgumentParser(description="Generate tilelang flash attention kernels")
    parser.add_argument("--output-dir", default="../generated", help="Output directory")
    parser.add_argument("--target", choices=["cuda", "hip"], default="hip",
                        help="Target backend (hip for AMD)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gemma-4 e2b configs to generate
    configs = [
        {"name": "flash_attn_gqa_256_causal", "desc": "GQA flash attn head_dim=256 causal",
         "params": {"batch": 1, "heads_q": 8, "heads_kv": 1, "seq_q": 512, "seq_kv": 512,
                    "dim": 256, "is_causal": True}},
    ]

    manifest = {}
    for cfg in configs:
        name = cfg["name"]
        print(f"Generating {name}: {cfg['desc']}")

        try:
            kernel_func = make_flash_attn_gqa_256(**cfg["params"])

            # Compile to get kernel source
            ext = "hip" if args.target == "hip" else "cu"
            kernel = tilelang.compile(
                kernel_func,
                target=args.target,
                execution_backend="tvm_ffi",
            )
            source = kernel.get_kernel_source()

            filename = f"tl_{name}.{ext}"
            filepath = output_dir / filename
            filepath.write_text(source, encoding="utf-8")
            print(f"  -> {filepath} ({len(source)} bytes)")

            manifest[name] = {
                "file": filename,
                "desc": cfg["desc"],
                "target": args.target,
                "params": cfg["params"],
            }
        except Exception as e:
            print(f"  FAILED: {e}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nGenerated {len(manifest)} kernels. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
