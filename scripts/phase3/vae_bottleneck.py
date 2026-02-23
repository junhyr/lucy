#!/usr/bin/env python3
"""
Phase 3.2: VAE Bottleneck Analysis
Measure how VAE encode/decode time scales with resolution.
Test VAE tiling for high resolutions.

Usage:
    python scripts/phase3/vae_bottleneck.py --config configs/experiment_configs.yaml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import torch
import time
from omegaconf import OmegaConf

from utils.profiler import CUDAProfiler, get_gpu_memory_info
from scripts.phase1.run_baseline import build_streamdiffv2_config


def profile_vae_isolated(exp_cfg, checkpoint_folder=None):
    """Profile VAE encode/decode independently across resolutions."""
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    p3 = exp_cfg.phase3
    paths = exp_cfg.paths
    prof_cfg = exp_cfg.profiling

    sys.path.insert(0, paths.streamdiffv2)

    # Load just the VAE
    from causvid.models import get_vae_wrapper
    model_type = exp_cfg.model.model_type

    print("Loading VAE...")
    vae = get_vae_wrapper(model_name="wan")(model_type=model_type)
    vae = vae.to(device=device, dtype=torch.bfloat16)
    vae.eval()
    print("VAE loaded.")

    profiler = CUDAProfiler(
        num_warmup=prof_cfg.num_warmup,
        num_iterations=prof_cfg.num_iterations,
        sync=True,
    )

    results = {}
    chunk_frames = 4  # Typical chunk: 4 frames

    for res in p3.resolutions:
        name = res["name"]
        h, w = res["height"], res["width"]
        print(f"\n--- VAE @ {name} ({h}x{w}) ---")

        profiler.reset()

        try:
            # Create dummy input (pixel space): [B, C, T, H, W]
            dummy_video = torch.randn(1, 3, chunk_frames, h, w, device=device, dtype=torch.bfloat16)

            # Warmup
            for _ in range(prof_cfg.num_warmup):
                with torch.no_grad():
                    latents = vae.stream_encode(dummy_video)
                    _ = vae.stream_decode_to_pixel(latents.transpose(2, 1))
                torch.cuda.synchronize()

            # Reset VAE internal caches if any
            if hasattr(vae, 'reset_cache'):
                vae.reset_cache()

            # Measure encode
            encode_times = []
            for _ in range(prof_cfg.num_iterations):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.no_grad():
                    latents = vae.stream_encode(dummy_video)
                end.record()
                torch.cuda.synchronize()
                encode_times.append(start.elapsed_time(end))

            # Measure decode
            decode_times = []
            latents_for_decode = latents.transpose(2, 1).contiguous()
            for _ in range(prof_cfg.num_iterations):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.no_grad():
                    _ = vae.stream_decode_to_pixel(latents_for_decode)
                end.record()
                torch.cuda.synchronize()
                decode_times.append(start.elapsed_time(end))

            avg_encode = sum(encode_times) / len(encode_times)
            avg_decode = sum(decode_times) / len(decode_times)

            # Estimate VAE memory
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = vae.stream_encode(dummy_video)
                _ = vae.stream_decode_to_pixel(latents_for_decode)
            mem = get_gpu_memory_info()

            results[name] = {
                "resolution": f"{h}x{w}",
                "chunk_frames": chunk_frames,
                "encode_avg_ms": round(avg_encode, 2),
                "decode_avg_ms": round(avg_decode, 2),
                "total_vae_ms": round(avg_encode + avg_decode, 2),
                "latent_shape": list(latents.shape),
                "peak_memory_mb": mem["max_allocated_mb"],
            }

            print(f"  Encode: {avg_encode:.2f} ms")
            print(f"  Decode: {avg_decode:.2f} ms")
            print(f"  Total:  {avg_encode + avg_decode:.2f} ms")
            print(f"  Latent shape: {latents.shape}")

        except torch.cuda.OutOfMemoryError:
            results[name] = {
                "resolution": f"{h}x{w}",
                "error": "OutOfMemoryError",
            }
            print(f"  OOM!")

        except Exception as e:
            results[name] = {
                "resolution": f"{h}x{w}",
                "error": str(e),
            }
            print(f"  ERROR: {e}")

        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"VAE BOTTLENECK ANALYSIS")
    print(f"{'='*70}")
    print(f"{'Resolution':<12} {'Encode ms':>10} {'Decode ms':>10} {'Total ms':>10} {'Peak MB':>10}")
    print(f"{'─'*55}")
    for name, r in results.items():
        if "error" in r:
            print(f"{r['resolution']:<12} {'ERROR':>10}")
        else:
            print(f"{r['resolution']:<12} {r['encode_avg_ms']:>10.2f} {r['decode_avg_ms']:>10.2f} {r['total_vae_ms']:>10.2f} {r['peak_memory_mb']:>10.0f}")

    # Scaling analysis
    valid = [(r["resolution"], r["total_vae_ms"]) for r in results.values() if "error" not in r]
    if len(valid) >= 2:
        first, last = valid[0], valid[-1]
        pixel_ratio = 1  # approximate
        time_ratio = last[1] / first[1] if first[1] > 0 else 0
        print(f"\nScaling: {first[0]} → {last[0]}: {time_ratio:.1f}x time increase")

    out_path = os.path.join(paths.output_root, "phase3", "vae_bottleneck", "vae_results.json")
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
    profile_vae_isolated(exp_cfg, args.checkpoint_folder)


if __name__ == "__main__":
    main()
