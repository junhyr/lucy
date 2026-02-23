#!/usr/bin/env python3
"""
Phase 2.1: torch.compile Effect on DiT
Apply torch.compile to the full DiT model and measure fps improvement.

Usage:
    python scripts/phase2/opt_torch_compile.py --config configs/experiment_configs.yaml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import torch
import time
from omegaconf import OmegaConf

from utils.profiler import CUDAProfiler, FPSTracker, get_gpu_memory_info
from scripts.phase1.run_baseline import build_streamdiffv2_config


def run_compile_experiment(exp_cfg, checkpoint_folder=None):
    """Test torch.compile on the full DiT model."""
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    p1 = exp_cfg.phase1
    p2_compile = exp_cfg.phase2.compile
    paths = exp_cfg.paths
    prof_cfg = exp_cfg.profiling

    sys.path.insert(0, paths.streamdiffv2)
    from streamv2v.inference import SingleGPUInferencePipeline

    cfg = build_streamdiffv2_config(exp_cfg)
    ckpt_folder = checkpoint_folder or os.path.join(paths.checkpoint_folder, "causvid")

    results = {}

    for compile_mode in [None, "default", "reduce-overhead", "max-autotune"]:
        label = compile_mode or "no_compile"
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"{'='*60}")

        # Re-initialize to get clean state
        pipeline_mgr = SingleGPUInferencePipeline(cfg, device)
        pipeline_mgr.load_model(ckpt_folder)

        if compile_mode is not None:
            print(f"Applying torch.compile(mode='{compile_mode}')...")
            model = pipeline_mgr.pipeline.generator.model
            pipeline_mgr.pipeline.generator.model = torch.compile(
                model,
                mode=compile_mode,
                dynamic=p2_compile.dynamic,
                fullgraph=p2_compile.fullgraph,
            )
            # Warmup compilation with dummy forward
            print("Warming up compiled model...")

        # Run inference
        chunk_size = p1.chunk_size * cfg.num_frame_per_block
        total_chunks = prof_cfg.num_warmup + prof_cfg.num_iterations
        num_steps = len(cfg.denoising_step_list)
        prompts = [p1.prompt]

        output_dir = os.path.join(paths.output_root, "phase2", "compile", label)
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

        results[label] = {
            "compile_mode": label,
            "total_frames": total_frames,
            "total_time_s": round(total_time, 2),
            "avg_fps": round(total_frames / total_time, 2),
            "avg_ms_per_frame": round(total_time / total_frames * 1000, 2),
            "gpu_memory": get_gpu_memory_info(),
        }

        print(f"  FPS: {results[label]['avg_fps']:.2f}  |  ms/frame: {results[label]['avg_ms_per_frame']:.2f}")

        # Clean up
        del pipeline_mgr
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"TORCH.COMPILE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Mode':<20} {'FPS':>8} {'ms/frame':>10} {'Speedup':>8}")
    print(f"{'─'*50}")
    baseline_fps = results.get("no_compile", {}).get("avg_fps", 1)
    for label, r in results.items():
        speedup = r["avg_fps"] / baseline_fps if baseline_fps > 0 else 0
        print(f"{label:<20} {r['avg_fps']:>8.2f} {r['avg_ms_per_frame']:>10.2f} {speedup:>7.2f}x")

    # Save
    out_path = os.path.join(paths.output_root, "phase2", "compile", "compile_results.json")
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
    run_compile_experiment(exp_cfg, args.checkpoint_folder)


if __name__ == "__main__":
    main()
