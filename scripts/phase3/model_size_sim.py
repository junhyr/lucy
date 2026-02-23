#!/usr/bin/env python3
"""
Phase 3.3: Model Size Simulation
Extrapolate from 1.3B per-block measurements to estimate 3B and 5B performance.

Approach:
  1. Measure individual block forward time at the 1.3B scale (dim=1536, 12 heads)
  2. Create synthetic blocks at larger dims (2048, 2560) and measure them
  3. Build the resolution × model_size fps matrix

Usage:
    python scripts/phase3/model_size_sim.py --config configs/experiment_configs.yaml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import torch
import torch.nn as nn
import time
from omegaconf import OmegaConf

from utils.profiler import CUDAProfiler, get_gpu_memory_info, estimate_kv_cache_memory_mb


def create_synthetic_attention_block(dim, num_heads, device, dtype=torch.bfloat16):
    """Create a minimal synthetic DiT attention block for timing."""
    head_dim = dim // num_heads

    class SyntheticBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.o_proj = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim)
            # Cross-attention
            self.cross_q = nn.Linear(dim, dim)
            self.cross_k = nn.Linear(dim, dim)
            self.cross_v = nn.Linear(dim, dim)
            self.cross_o = nn.Linear(dim, dim)
            self.norm3 = nn.LayerNorm(dim)
            # FFN (typical 4x expansion)
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )
            # Modulation (6 vectors as in CausalWanAttentionBlock)
            self.modulation = nn.Linear(dim, dim * 6)

        def forward(self, x, context=None):
            # Self-attention
            residual = x
            x_norm = self.norm1(x)
            q = self.q_proj(x_norm).reshape(x.shape[0], -1, num_heads, head_dim)
            k = self.k_proj(x_norm).reshape(x.shape[0], -1, num_heads, head_dim)
            v = self.v_proj(x_norm).reshape(x.shape[0], -1, num_heads, head_dim)

            # Scaled dot-product attention (uses FlashAttention on H100)
            q = q.transpose(1, 2)  # B, H, S, D
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attn_out = attn_out.transpose(1, 2).reshape(x.shape[0], -1, dim)

            x = residual + self.o_proj(attn_out)

            # Cross-attention
            if context is not None:
                residual = x
                x_norm = self.norm2(x)
                q = self.cross_q(x_norm).reshape(x.shape[0], -1, num_heads, head_dim).transpose(1, 2)
                k = self.cross_k(context).reshape(x.shape[0], -1, num_heads, head_dim).transpose(1, 2)
                v = self.cross_v(context).reshape(x.shape[0], -1, num_heads, head_dim).transpose(1, 2)
                attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                attn_out = attn_out.transpose(1, 2).reshape(x.shape[0], -1, dim)
                x = residual + self.cross_o(attn_out)

            # FFN
            residual = x
            x = residual + self.ffn(self.norm3(x))

            return x

    block = SyntheticBlock().to(device=device, dtype=dtype).eval()
    return block


def measure_block_time(block, seq_len, dim, device, num_warmup=5, num_iters=20):
    """Measure average forward time of a single block in ms."""
    dtype = torch.bfloat16
    x = torch.randn(1, seq_len, dim, device=device, dtype=dtype)
    context = torch.randn(1, 512, dim, device=device, dtype=dtype)  # text context

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = block(x, context)
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            _ = block(x, context)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


def run_model_size_simulation(exp_cfg, checkpoint_folder=None):
    """Run model size simulation using synthetic blocks."""
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    p3 = exp_cfg.phase3
    model_cfg = exp_cfg.model
    prof_cfg = exp_cfg.profiling

    print("=== Phase 3.3: Model Size Simulation ===")
    print("Measuring per-block forward time at different dims and resolutions")
    print()

    # Collect: per-block time for each (dim, seq_len) combination
    model_specs = p3.model_size_simulation
    resolutions = p3.resolutions

    results = {
        "per_block_times": {},    # block_time[model_name][resolution] = ms
        "estimated_fps": {},      # fps[model_name][resolution] = fps
    }

    for spec in model_specs:
        model_name = spec["name"]
        dim = spec["dim"]
        heads = spec["heads"]
        num_blocks = spec["blocks"]

        print(f"\n--- Model: {model_name} (dim={dim}, heads={heads}, blocks={num_blocks}) ---")

        block = create_synthetic_attention_block(dim, heads, device)
        block_times = {}

        for res in resolutions:
            res_name = res["name"]
            h, w = res["height"], res["width"]
            seq_len = (h // 16) * (w // 16)  # frame_seq_length

            print(f"  {res_name} (seq_len={seq_len})...", end=" ", flush=True)

            try:
                block_ms = measure_block_time(
                    block, seq_len, dim, device,
                    num_warmup=prof_cfg.num_warmup,
                    num_iters=prof_cfg.num_iterations,
                )
                total_dit_ms = block_ms * num_blocks
                chunk_size = 4  # frames per chunk
                # Rough estimate: DiT is ~70-80% of total, VAE is rest
                estimated_total_ms = total_dit_ms / 0.75
                estimated_fps = chunk_size / (estimated_total_ms / 1000)

                block_times[res_name] = {
                    "seq_len": seq_len,
                    "block_ms": round(block_ms, 3),
                    "total_dit_ms": round(total_dit_ms, 2),
                    "estimated_total_ms": round(estimated_total_ms, 2),
                    "estimated_fps": round(estimated_fps, 2),
                }

                # KV cache memory
                kv_mem = estimate_kv_cache_memory_mb(num_blocks, seq_len * 6, heads)
                block_times[res_name]["kv_cache_mb"] = round(kv_mem, 1)

                print(f"block={block_ms:.2f}ms, dit={total_dit_ms:.1f}ms, est_fps={estimated_fps:.1f}")

            except torch.cuda.OutOfMemoryError:
                block_times[res_name] = {"error": "OOM", "seq_len": seq_len}
                print("OOM")
            except Exception as e:
                block_times[res_name] = {"error": str(e), "seq_len": seq_len}
                print(f"ERROR: {e}")

            torch.cuda.empty_cache()

        results["per_block_times"][model_name] = block_times

        del block
        torch.cuda.empty_cache()

    # ── Build fps matrix ──
    print(f"\n{'='*70}")
    print(f"RESOLUTION x MODEL SIZE FPS MATRIX (1xH100, estimated, 2-step)")
    print(f"{'='*70}")

    # Header
    model_names = [s["name"] for s in model_specs]
    header = f"{'Resolution':<12}" + "".join(f"{n:>12}" for n in model_names)
    print(header)
    print("─" * (12 + 12 * len(model_names)))

    for res in resolutions:
        rn = res["name"]
        row = f"{rn:<12}"
        for mn in model_names:
            bt = results["per_block_times"].get(mn, {}).get(rn, {})
            if "error" in bt:
                row += f"{'OOM':>12}"
            elif "estimated_fps" in bt:
                row += f"{bt['estimated_fps']:>11.1f}f"
            else:
                row += f"{'?':>12}"
        print(row)

    # Save
    paths = exp_cfg.paths
    out_path = os.path.join(paths.output_root, "phase3", "model_size", "model_size_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_configs.yaml")
    parser.add_argument("--checkpoint_folder", type=str, default=None)
    args = parser.parse_args()
    exp_cfg = OmegaConf.load(args.config)
    run_model_size_simulation(exp_cfg, args.checkpoint_folder)


if __name__ == "__main__":
    main()
