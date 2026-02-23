#!/usr/bin/env python3
"""
Phase 4.3: Throughput vs Latency Separation
Measure:
  - Throughput: sustained fps over 100 frames
  - Latency: input→output time for a single frame
  - Pipeline warmup: frames until stable throughput
  - Jitter: frame-to-frame time variance

Usage:
    # Single GPU
    python scripts/phase4/throughput_latency.py --config configs/experiment_configs.yaml --num_gpus 1

    # Multi GPU
    torchrun --nproc_per_node=4 scripts/phase4/throughput_latency.py --config configs/experiment_configs.yaml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import torch
import time
import statistics
from omegaconf import OmegaConf

from utils.profiler import FPSTracker, get_gpu_memory_info
from scripts.phase1.run_baseline import build_streamdiffv2_config


def measure_throughput_latency_single_gpu(exp_cfg, checkpoint_folder=None):
    """Throughput and latency on single GPU."""
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    p1 = exp_cfg.phase1
    p4 = exp_cfg.phase4
    paths = exp_cfg.paths

    sys.path.insert(0, paths.streamdiffv2)
    from streamv2v.inference import SingleGPUInferencePipeline
    from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline

    cfg = build_streamdiffv2_config(exp_cfg)
    ckpt_folder = checkpoint_folder or os.path.join(paths.checkpoint_folder, "causvid")

    print("=== Throughput vs Latency (Single GPU) ===")

    pipeline_mgr = SingleGPUInferencePipeline(cfg, device)
    pipeline_mgr.load_model(ckpt_folder)

    chunk_size = p1.chunk_size * cfg.num_frame_per_block
    num_steps = len(cfg.denoising_step_list)
    prompts = [p1.prompt]

    # Initialize pipeline
    noise = torch.randn(
        1, 1 + cfg.num_frame_per_block, 16,
        p1.height // 16 * 2, p1.width // 16 * 2,
        device=device, dtype=torch.bfloat16
    )

    frame_seq_length = pipeline_mgr.pipeline.frame_seq_length
    current_start = 0
    current_end = frame_seq_length * (1 + chunk_size // 4)

    pipeline_mgr.prepare_pipeline(
        text_prompts=prompts,
        noise=noise,
        current_start=current_start,
        current_end=current_end,
    )

    # ── Throughput measurement ──
    print(f"\nMeasuring throughput ({p4.throughput_test.num_frames} frames)...")
    total_frames = p4.throughput_test.num_frames
    warmup_frames = p4.throughput_test.warmup_frames

    per_chunk_times = []
    current_start = current_end

    for i in range(total_frames // chunk_size):
        current_end = current_start + (chunk_size // 4) * frame_seq_length

        chunk_noise = torch.randn(
            1, cfg.num_frame_per_block, 16,
            p1.height // 16 * 2, p1.width // 16 * 2,
            device=device, dtype=torch.bfloat16
        )

        torch.cuda.synchronize()
        start = time.perf_counter()

        # DiT inference
        denoised = pipeline_mgr.pipeline.inference_stream(
            noise=chunk_noise,
            current_start=current_start,
            current_end=current_end,
            current_step=None,
        )

        # VAE decode
        video = pipeline_mgr.pipeline.vae.stream_decode_to_pixel(denoised[[-1]])

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        per_chunk_times.append(elapsed)
        current_start = current_end

    # Separate warmup from measured
    measured_times = per_chunk_times[warmup_frames // chunk_size:]
    warmup_times = per_chunk_times[:warmup_frames // chunk_size]

    # ── Results ──
    throughput_fps = chunk_size / statistics.mean(measured_times) if measured_times else 0
    avg_latency_ms = statistics.mean(measured_times) * 1000 if measured_times else 0
    jitter_ms = statistics.stdev(measured_times) * 1000 if len(measured_times) > 1 else 0
    warmup_avg = statistics.mean(warmup_times) if warmup_times else 0

    # Find warmup convergence point
    warmup_convergence = 0
    if len(per_chunk_times) > 3:
        stable_time = statistics.mean(measured_times) if measured_times else per_chunk_times[-1]
        for i, t in enumerate(per_chunk_times):
            if abs(t - stable_time) / stable_time < 0.1:
                warmup_convergence = i
                break

    results = {
        "mode": "single_gpu",
        "throughput_fps": round(throughput_fps, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "min_latency_ms": round(min(measured_times) * 1000, 2) if measured_times else 0,
        "max_latency_ms": round(max(measured_times) * 1000, 2) if measured_times else 0,
        "jitter_ms": round(jitter_ms, 2),
        "warmup_convergence_chunk": warmup_convergence,
        "warmup_avg_ms": round(warmup_avg * 1000, 2),
        "per_chunk_times_ms": [round(t * 1000, 2) for t in per_chunk_times],
        "total_measured_frames": len(measured_times) * chunk_size,
    }

    print(f"\n{'='*60}")
    print(f"THROUGHPUT vs LATENCY (1xH100)")
    print(f"{'='*60}")
    print(f"Throughput:          {throughput_fps:.2f} fps")
    print(f"Avg latency:         {avg_latency_ms:.2f} ms ({chunk_size} frames/chunk)")
    print(f"Min latency:         {results['min_latency_ms']:.2f} ms")
    print(f"Max latency:         {results['max_latency_ms']:.2f} ms")
    print(f"Jitter (stddev):     {jitter_ms:.2f} ms")
    print(f"Warmup convergence:  chunk #{warmup_convergence}")
    print(f"30fps target:        {'YES' if throughput_fps >= 30 else 'NO'} (need {33.33:.1f} ms/chunk)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_configs.yaml")
    parser.add_argument("--checkpoint_folder", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    exp_cfg = OmegaConf.load(args.config)

    if args.num_gpus == 1:
        results = measure_throughput_latency_single_gpu(exp_cfg, args.checkpoint_folder)
    else:
        # Multi-GPU version would use torchrun
        print("For multi-GPU, use: torchrun --nproc_per_node=N scripts/phase4/multi_gpu_baseline.py")
        return

    # Save
    paths = exp_cfg.paths
    out_path = os.path.join(paths.output_root, "phase4", "throughput_latency",
                           f"throughput_latency_{args.num_gpus}gpu.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
