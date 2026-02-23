#!/usr/bin/env python3
"""
Phase 1: Baseline Inference — Run StreamDiffusionV2 SingleGPUInferencePipeline
and measure end-to-end fps at 480p on 1xH100.

Usage:
    python scripts/phase1/run_baseline.py --config configs/experiment_configs.yaml
    python scripts/phase1/run_baseline.py --config configs/experiment_configs.yaml --video_path /path/to/input.mp4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import time
import torch
import numpy as np
from omegaconf import OmegaConf

from utils.profiler import CUDAProfiler, FPSTracker, get_gpu_memory_info


def build_streamdiffv2_config(exp_cfg):
    """Build a config compatible with CausalStreamInferencePipeline from our experiment config."""
    p1 = exp_cfg.phase1
    model = exp_cfg.model

    # Load the base StreamDiffusionV2 YAML config
    base_config_path = model.config_path
    if os.path.exists(base_config_path):
        cfg = OmegaConf.load(base_config_path)
    else:
        cfg = OmegaConf.create()

    # Override with our settings
    cfg.height = p1.height
    cfg.width = p1.width
    cfg.model_type = model.model_type

    # Denoising steps
    step_map = {1: [700, 0], 2: [700, 500, 0], 3: [700, 600, 400, 0], 4: [700, 600, 500, 400, 0]}
    cfg.denoising_step_list = step_map.get(p1.denoising_steps, [700, 500, 0])

    # Ensure required fields
    cfg.setdefault("num_kv_cache", 6)
    cfg.setdefault("num_sink_tokens", 3)
    cfg.setdefault("adapt_sink_threshold", 0.2)
    cfg.setdefault("num_frame_per_block", 1)
    cfg.setdefault("warp_denoising_step", False)
    cfg.setdefault("generator_name", "causal_wan")
    cfg.setdefault("model_name", "wan")
    cfg.setdefault("num_train_timestep", 1000)
    cfg.setdefault("timestep_shift", 8.0)

    return cfg


def run_baseline(exp_cfg, video_path=None, checkpoint_folder=None):
    """Run baseline inference and collect timing metrics."""
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    p1 = exp_cfg.phase1
    paths = exp_cfg.paths

    # Import StreamDiffusionV2 components
    sys.path.insert(0, paths.streamdiffv2)
    from streamv2v.inference import SingleGPUInferencePipeline, load_mp4_as_tensor

    # Build config
    cfg = build_streamdiffv2_config(exp_cfg)
    ckpt_folder = checkpoint_folder or os.path.join(paths.checkpoint_folder, "causvid")

    print(f"=== Phase 1: Baseline Inference ===")
    print(f"Resolution: {p1.height}x{p1.width}")
    print(f"Denoising steps: {p1.denoising_steps} → {cfg.denoising_step_list}")
    print(f"Chunk size: {p1.chunk_size}")
    print(f"Checkpoint: {ckpt_folder}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Initialize pipeline
    print("Loading model...")
    mem_before = get_gpu_memory_info()
    pipeline = SingleGPUInferencePipeline(cfg, device)
    pipeline.load_model(ckpt_folder)
    mem_after = get_gpu_memory_info()
    print(f"Model loaded. GPU memory: {mem_before['allocated_mb']:.0f}MB → {mem_after['allocated_mb']:.0f}MB")
    print()

    # Load or generate input
    chunk_size = p1.chunk_size * cfg.num_frame_per_block
    num_chunks = p1.num_warmup_chunks + p1.num_measure_chunks

    if video_path and os.path.exists(video_path):
        print(f"Loading input video: {video_path}")
        input_video = load_mp4_as_tensor(
            video_path,
            max_frames=1 + chunk_size * (num_chunks + len(cfg.denoising_step_list)),
            resize_hw=(p1.height, p1.width),
        ).unsqueeze(0).to(dtype=torch.bfloat16, device=device)
        print(f"Input tensor shape: {input_video.shape}")
    else:
        print("No input video — running T2V mode with random noise")
        input_video = None

    # Prepare prompts
    prompts = [p1.prompt]
    num_steps = len(cfg.denoising_step_list)

    # Create output dir
    output_dir = os.path.join(paths.output_root, "phase1", "baseline")
    os.makedirs(output_dir, exist_ok=True)

    # ── Run inference with timing ──────────────────────
    print(f"\nRunning inference: {num_chunks} chunks ({p1.num_warmup_chunks} warmup + {p1.num_measure_chunks} measured)...")

    fps_tracker = FPSTracker(warmup_frames=p1.num_warmup_chunks * chunk_size)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    pipeline.run_inference(
        input_video_original=input_video,
        prompts=prompts,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        noise_scale=p1.noise_scale,
        output_folder=output_dir,
        fps=p1.fps,
        target_fps=None,
        num_steps=num_steps,
    )

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    total_frames = num_chunks * chunk_size

    # ── Collect results ────────────────────────────────
    results = {
        "phase": "phase1_baseline",
        "resolution": f"{p1.height}x{p1.width}",
        "model": exp_cfg.model.model_type,
        "denoising_steps": p1.denoising_steps,
        "chunk_size": chunk_size,
        "total_chunks": num_chunks,
        "total_frames": total_frames,
        "total_time_s": round(total_time, 2),
        "avg_fps": round(total_frames / total_time, 2),
        "avg_ms_per_frame": round(total_time / total_frames * 1000, 2),
        "gpu_memory": get_gpu_memory_info(),
        "gpu": torch.cuda.get_device_name(0),
    }

    # Save results
    results_path = os.path.join(output_dir, "baseline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"Resolution:      {results['resolution']}")
    print(f"Model:           {results['model']}")
    print(f"Steps:           {results['denoising_steps']}")
    print(f"Total frames:    {results['total_frames']}")
    print(f"Total time:      {results['total_time_s']:.2f}s")
    print(f"Average FPS:     {results['avg_fps']:.2f}")
    print(f"ms/frame:        {results['avg_ms_per_frame']:.2f}")
    print(f"Peak GPU mem:    {results['gpu_memory']['max_allocated_mb']:.0f}MB")
    print(f"Results saved:   {results_path}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline Inference")
    parser.add_argument("--config", type=str, default="configs/experiment_configs.yaml")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--checkpoint_folder", type=str, default=None)
    args = parser.parse_args()

    exp_cfg = OmegaConf.load(args.config)
    run_baseline(exp_cfg, video_path=args.video_path, checkpoint_folder=args.checkpoint_folder)


if __name__ == "__main__":
    main()
