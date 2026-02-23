#!/usr/bin/env python3
"""
Phase 3.1: Resolution Scaling Experiment
Measure fps at 360p → 480p → 544p → 720p → 1080p.
Uses the best optimization settings from Phase 2.

Usage:
    python scripts/phase3/resolution_scaling.py --config configs/experiment_configs.yaml
    python scripts/phase3/resolution_scaling.py --config configs/experiment_configs.yaml --optimizations compile
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import torch
import time
from omegaconf import OmegaConf

from utils.profiler import CUDAProfiler, get_gpu_memory_info, estimate_kv_cache_memory_mb
from scripts.phase1.run_baseline import build_streamdiffv2_config


def run_resolution_scaling(exp_cfg, checkpoint_folder=None, apply_compile=False):
    """Test multiple resolutions with the same model."""
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    p1 = exp_cfg.phase1
    p3 = exp_cfg.phase3
    paths = exp_cfg.paths
    prof_cfg = exp_cfg.profiling
    model_cfg = exp_cfg.model

    sys.path.insert(0, os.path.join(paths.streamdiffv2, ".."))
    from StreamDiffusionV2.streamv2v.inference import SingleGPUInferencePipeline

    ckpt_folder = checkpoint_folder or os.path.join(paths.checkpoint_folder, "causvid")
    prompts = [p1.prompt]

    results = {}

    for res in p3.resolutions:
        name = res["name"]
        height = res["height"]
        width = res["width"]

        print(f"\n{'='*60}")
        print(f"Testing resolution: {name} ({height}x{width})")
        print(f"{'='*60}")

        # Calculate frame_seq_length and KV cache size
        frame_seq_length = (height // 16) * (width // 16)
        kv_cache_mem = estimate_kv_cache_memory_mb(
            num_blocks=model_cfg.num_transformer_blocks,
            cache_length=frame_seq_length * p1.get("num_kv_cache", 6),
            num_heads=model_cfg.num_heads,
        )
        print(f"  frame_seq_length: {frame_seq_length}")
        print(f"  Est. KV cache: {kv_cache_mem:.1f} MB")

        # Build config for this resolution
        cfg = build_streamdiffv2_config(exp_cfg)
        cfg.height = height
        cfg.width = width

        try:
            pipeline_mgr = SingleGPUInferencePipeline(cfg, device)
            pipeline_mgr.load_model(ckpt_folder)

            if apply_compile:
                print("  Applying torch.compile...")
                pipeline_mgr.pipeline.generator.model = torch.compile(
                    pipeline_mgr.pipeline.generator.model,
                    mode="max-autotune",
                    dynamic=False,
                )

            chunk_size = p1.chunk_size * cfg.num_frame_per_block
            total_chunks = prof_cfg.num_warmup + prof_cfg.num_iterations
            num_steps = len(cfg.denoising_step_list)

            output_dir = os.path.join(paths.output_root, "phase3", "resolution", name)
            os.makedirs(output_dir, exist_ok=True)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()

            pipeline_mgr.run_inference(
                input_video_original=None,
                prompts=prompts,
                num_chunks=total_chunks,
                chunk_size=chunk_size,
                noise_scale=p1.noise_scale,
                output_folder=output_dir,
                fps=p1.fps,
                target_fps=None,
                num_steps=num_steps,
            )

            torch.cuda.synchronize()
            total_time = time.perf_counter() - start
            total_frames = total_chunks * chunk_size

            mem_info = get_gpu_memory_info()

            results[name] = {
                "resolution": f"{height}x{width}",
                "frame_seq_length": frame_seq_length,
                "kv_cache_memory_mb": round(kv_cache_mem, 1),
                "total_frames": total_frames,
                "total_time_s": round(total_time, 2),
                "avg_fps": round(total_frames / total_time, 2),
                "avg_ms_per_frame": round(total_time / total_frames * 1000, 2),
                "peak_memory_mb": mem_info["max_allocated_mb"],
                "gpu_memory": mem_info,
            }

            print(f"  FPS: {results[name]['avg_fps']:.2f}")
            print(f"  Peak memory: {mem_info['max_allocated_mb']:.0f} MB")

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at {name}!")
            results[name] = {
                "resolution": f"{height}x{width}",
                "frame_seq_length": frame_seq_length,
                "error": "OutOfMemoryError",
                "kv_cache_memory_mb": round(kv_cache_mem, 1),
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {
                "resolution": f"{height}x{width}",
                "error": str(e),
            }
        finally:
            if 'pipeline_mgr' in dir():
                del pipeline_mgr
            torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"RESOLUTION SCALING RESULTS (1xH100, {model_cfg.model_type})")
    print(f"{'='*70}")
    print(f"{'Resolution':<12} {'SeqLen':>8} {'KV MB':>8} {'FPS':>8} {'ms/frame':>10} {'Peak MB':>10}")
    print(f"{'─'*60}")

    for name, r in results.items():
        if "error" in r:
            print(f"{r['resolution']:<12} {r.get('frame_seq_length', '?'):>8} {r.get('kv_cache_memory_mb', '?'):>8} {'OOM/ERR':>8}")
        else:
            print(f"{r['resolution']:<12} {r['frame_seq_length']:>8} {r['kv_cache_memory_mb']:>8.1f} {r['avg_fps']:>8.2f} {r['avg_ms_per_frame']:>10.2f} {r['peak_memory_mb']:>10.0f}")

    # Find OOM boundary
    oom_resolutions = [n for n, r in results.items() if "error" in r and "OutOfMemory" in r.get("error", "")]
    if oom_resolutions:
        print(f"\nOOM boundary: {oom_resolutions[0]}")

    out_path = os.path.join(paths.output_root, "phase3", "resolution", "resolution_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_configs.yaml")
    parser.add_argument("--checkpoint_folder", type=str, default=None)
    parser.add_argument("--compile", action="store_true", help="Apply torch.compile")
    args = parser.parse_args()
    exp_cfg = OmegaConf.load(args.config)
    run_resolution_scaling(exp_cfg, args.checkpoint_folder, apply_compile=args.compile)


if __name__ == "__main__":
    main()
