#!/usr/bin/env python3
"""
Phase 2.5: Denoising Step Count Experiment
Compare 1-step, 2-step, 3-step, 4-step inference.
Measure fps and qualitative impact.

Usage:
    python scripts/phase2/opt_denoising_steps.py --config configs/experiment_configs.yaml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import torch
import time
from omegaconf import OmegaConf

from utils.profiler import get_gpu_memory_info
from scripts.phase1.run_baseline import build_streamdiffv2_config


def run_step_experiment(exp_cfg, checkpoint_folder=None, video_path=None):
    """Test different denoising step counts."""
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    p1 = exp_cfg.phase1
    paths = exp_cfg.paths
    prof_cfg = exp_cfg.profiling
    step_variants = exp_cfg.phase2.denoising_steps.variants

    sys.path.insert(0, paths.streamdiffv2)
    from streamv2v.inference import SingleGPUInferencePipeline, load_mp4_as_tensor

    ckpt_folder = checkpoint_folder or os.path.join(paths.checkpoint_folder, "causvid")

    # Load input video once if available
    input_video = None
    if video_path and os.path.exists(video_path):
        input_video = load_mp4_as_tensor(
            video_path,
            max_frames=200,  # Enough for all step variants
            resize_hw=(p1.height, p1.width),
        ).unsqueeze(0).to(dtype=torch.bfloat16, device=device)

    results = {}

    for variant in step_variants:
        name = variant["name"]
        steps = list(variant["steps"])

        print(f"\n{'='*60}")
        print(f"Testing: {name} — steps={steps}")
        print(f"{'='*60}")

        # Build config with this step list
        cfg = build_streamdiffv2_config(exp_cfg)
        cfg.denoising_step_list = steps

        pipeline_mgr = SingleGPUInferencePipeline(cfg, device)
        pipeline_mgr.load_model(ckpt_folder)

        chunk_size = p1.chunk_size * cfg.num_frame_per_block
        total_chunks = prof_cfg.num_warmup + prof_cfg.num_iterations
        num_steps = len(steps)
        prompts = [p1.prompt]

        output_dir = os.path.join(paths.output_root, "phase2", "denoising_steps", name)
        os.makedirs(output_dir, exist_ok=True)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()

        try:
            pipeline_mgr.run_inference(
                input_video_original=input_video,
                prompts=prompts,
                num_chunks=total_chunks,
                chunk_size=chunk_size,
                noise_scale=p1.noise_scale,
                output_folder=output_dir,
                fps=p1.fps,
                target_fps=None,
                num_steps=num_steps,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e)}
            del pipeline_mgr
            torch.cuda.empty_cache()
            continue

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        total_frames = total_chunks * chunk_size

        results[name] = {
            "denoising_steps": steps,
            "num_steps": num_steps,
            "total_frames": total_frames,
            "total_time_s": round(total_time, 2),
            "avg_fps": round(total_frames / total_time, 2),
            "avg_ms_per_frame": round(total_time / total_frames * 1000, 2),
            "gpu_memory": get_gpu_memory_info(),
            "output_video": os.path.join(output_dir, "output_000.mp4"),
        }

        print(f"  FPS: {results[name]['avg_fps']:.2f}  |  ms/frame: {results[name]['avg_ms_per_frame']:.2f}")

        del pipeline_mgr
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"DENOISING STEPS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Variant':<15} {'Steps':>6} {'FPS':>8} {'ms/frame':>10} {'Quality':>10}")
    print(f"{'─'*55}")

    # Baseline reference for relative quality note
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<15} {'ERR':>6}")
            continue
        quality_note = "baseline" if r["num_steps"] == 3 else ("lower?" if r["num_steps"] < 3 else "higher?")
        print(f"{name:<15} {r['num_steps']:>6} {r['avg_fps']:>8.2f} {r['avg_ms_per_frame']:>10.2f} {quality_note:>10}")

    out_path = os.path.join(paths.output_root, "phase2", "denoising_steps", "step_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print(f"Output videos saved in: {os.path.join(paths.output_root, 'phase2', 'denoising_steps')}/")
    print("Review videos visually for quality comparison.")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_configs.yaml")
    parser.add_argument("--checkpoint_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    args = parser.parse_args()
    exp_cfg = OmegaConf.load(args.config)
    run_step_experiment(exp_cfg, args.checkpoint_folder, args.video_path)


if __name__ == "__main__":
    main()
