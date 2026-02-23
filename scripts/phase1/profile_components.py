#!/usr/bin/env python3
"""
Phase 1: Component-Level Profiling
Measure each pipeline component with torch.cuda.Event precision:
  - VAE stream_encode
  - DiT forward pass (overall + per-block-group)
  - VAE stream_decode
  - KV cache update
  - Total per chunk → fps

Usage:
    python scripts/phase1/profile_components.py --config configs/experiment_configs.yaml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import torch
import time
import numpy as np
from omegaconf import OmegaConf
from functools import wraps

from utils.profiler import CUDAProfiler, FPSTracker, get_gpu_memory_info, estimate_kv_cache_memory_mb


def patch_pipeline_for_profiling(pipeline, profiler):
    """Monkey-patch the pipeline to insert CUDA event timing around key operations."""

    # ── Patch VAE encode ──
    original_stream_encode = pipeline.vae.stream_encode

    @wraps(original_stream_encode)
    def profiled_stream_encode(*args, **kwargs):
        profiler.start("vae_stream_encode")
        result = original_stream_encode(*args, **kwargs)
        profiler.stop("vae_stream_encode")
        return result

    pipeline.vae.stream_encode = profiled_stream_encode

    # ── Patch VAE decode ──
    original_stream_decode = pipeline.vae.stream_decode_to_pixel

    @wraps(original_stream_decode)
    def profiled_stream_decode(*args, **kwargs):
        profiler.start("vae_stream_decode")
        result = original_stream_decode(*args, **kwargs)
        profiler.stop("vae_stream_decode")
        return result

    pipeline.vae.stream_decode_to_pixel = profiled_stream_decode

    # ── Patch DiT forward (generator) ──
    original_generator_call = pipeline.generator.__class__.__call__

    @wraps(original_generator_call)
    def profiled_generator_call(self_gen, *args, **kwargs):
        profiler.start("dit_forward_total")
        result = original_generator_call(self_gen, *args, **kwargs)
        profiler.stop("dit_forward_total")
        return result

    pipeline.generator.__class__.__call__ = profiled_generator_call

    # ── Patch individual DiT blocks ──
    model = pipeline.generator.model
    if hasattr(model, 'blocks'):
        num_blocks = len(model.blocks)
        # Group blocks into 3 groups for measurement
        group_size = num_blocks // 3
        groups = [
            (0, group_size, "dit_blocks_0_to_9"),
            (group_size, 2 * group_size, "dit_blocks_10_to_19"),
            (2 * group_size, num_blocks, "dit_blocks_20_to_29"),
        ]

        for block_idx in range(num_blocks):
            block = model.blocks[block_idx]
            original_forward = block.forward
            # Find which group this block belongs to
            group_name = None
            for g_start, g_end, g_name in groups:
                if g_start <= block_idx < g_end:
                    group_name = g_name
                    break

            def make_profiled_forward(orig_fwd, grp_name, idx):
                @wraps(orig_fwd)
                def profiled_forward(*a, **kw):
                    profiler.start(f"dit_block_{idx}")
                    result = orig_fwd(*a, **kw)
                    profiler.stop(f"dit_block_{idx}")
                    return result
                return profiled_forward

            block.forward = make_profiled_forward(original_forward, group_name, block_idx)

    return pipeline


def run_profiling(exp_cfg, video_path=None, checkpoint_folder=None):
    """Run component-level profiling."""
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    p1 = exp_cfg.phase1
    prof_cfg = exp_cfg.profiling
    paths = exp_cfg.paths

    sys.path.insert(0, paths.streamdiffv2)
    from streamv2v.inference import SingleGPUInferencePipeline, load_mp4_as_tensor
    from scripts.phase1.run_baseline import build_streamdiffv2_config

    cfg = build_streamdiffv2_config(exp_cfg)
    ckpt_folder = checkpoint_folder or os.path.join(paths.checkpoint_folder, "causvid")

    print("=== Phase 1: Component Profiling ===")
    print(f"Resolution: {p1.height}x{p1.width}")
    print(f"Warmup iterations: {prof_cfg.num_warmup}")
    print(f"Measure iterations: {prof_cfg.num_iterations}")
    print()

    # Initialize
    profiler = CUDAProfiler(
        num_warmup=prof_cfg.num_warmup,
        num_iterations=prof_cfg.num_iterations,
        sync=prof_cfg.sync_before_measure,
    )

    pipeline_mgr = SingleGPUInferencePipeline(cfg, device)
    pipeline_mgr.load_model(ckpt_folder)

    # Patch for profiling
    patch_pipeline_for_profiling(pipeline_mgr.pipeline, profiler)

    # Prepare input
    chunk_size = p1.chunk_size * cfg.num_frame_per_block
    total_chunks = prof_cfg.num_warmup + prof_cfg.num_iterations

    if video_path and os.path.exists(video_path):
        input_video = load_mp4_as_tensor(
            video_path,
            max_frames=1 + chunk_size * (total_chunks + len(cfg.denoising_step_list)),
            resize_hw=(p1.height, p1.width),
        ).unsqueeze(0).to(dtype=torch.bfloat16, device=device)
    else:
        input_video = None

    prompts = [p1.prompt]
    num_steps = len(cfg.denoising_step_list)

    output_dir = os.path.join(paths.output_root, "phase1", "profiling")
    os.makedirs(output_dir, exist_ok=True)

    # Run inference (profiling hooks will capture timings)
    print("Running profiled inference...")
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

    # ── Aggregate block-group timings ──
    block_groups = {"dit_blocks_0_to_9": 0.0, "dit_blocks_10_to_19": 0.0, "dit_blocks_20_to_29": 0.0}
    block_group_counts = {"dit_blocks_0_to_9": 0, "dit_blocks_10_to_19": 0, "dit_blocks_20_to_29": 0}
    num_blocks = cfg.get("num_transformer_blocks", 30) if hasattr(cfg, "get") else 30
    group_size = num_blocks // 3

    for i in range(num_blocks):
        key = f"dit_block_{i}"
        if key in profiler.results:
            r = profiler.results[key]
            if i < group_size:
                grp = "dit_blocks_0_to_9"
            elif i < 2 * group_size:
                grp = "dit_blocks_10_to_19"
            else:
                grp = "dit_blocks_20_to_29"
            block_groups[grp] += r.gpu_avg_ms
            block_group_counts[grp] += 1

    # ── Build results ──
    frame_seq_length = (p1.height // 16) * (p1.width // 16)
    kv_cache_memory_mb = estimate_kv_cache_memory_mb(
        num_blocks=num_blocks,
        cache_length=frame_seq_length * cfg.num_kv_cache,
        num_heads=exp_cfg.model.num_heads,
    )

    component_breakdown = {}
    for key in ["vae_stream_encode", "dit_forward_total", "vae_stream_decode"]:
        if key in profiler.results:
            r = profiler.results[key]
            component_breakdown[key] = {"avg_ms": round(r.gpu_avg_ms, 2), "count": r.count}

    for grp, total_ms in block_groups.items():
        component_breakdown[grp] = {"avg_ms": round(total_ms, 2), "num_blocks": block_group_counts[grp]}

    # Per-chunk total
    total_per_chunk_ms = sum(
        component_breakdown.get(k, {}).get("avg_ms", 0)
        for k in ["vae_stream_encode", "dit_forward_total", "vae_stream_decode"]
    )
    fps_from_components = chunk_size / (total_per_chunk_ms / 1000) if total_per_chunk_ms > 0 else 0

    results = {
        "phase": "phase1_profiling",
        "resolution": f"{p1.height}x{p1.width}",
        "model": exp_cfg.model.model_type,
        "frame_seq_length": frame_seq_length,
        "kv_cache_memory_mb": round(kv_cache_memory_mb, 1),
        "chunk_size": chunk_size,
        "denoising_steps": p1.denoising_steps,
        "component_breakdown": component_breakdown,
        "total_per_chunk_ms": round(total_per_chunk_ms, 2),
        "estimated_fps": round(fps_from_components, 2),
        "gpu_memory": get_gpu_memory_info(),
        "all_timings": profiler.to_dict(),
    }

    # Save
    results_path = os.path.join(output_dir, "profiling_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print breakdown ──
    print(f"\n{'='*60}")
    print(f"COMPONENT PROFILING BREAKDOWN")
    print(f"{'='*60}")
    print(f"Resolution:           {results['resolution']}")
    print(f"Frame seq length:     {frame_seq_length}")
    print(f"KV cache memory:      {kv_cache_memory_mb:.1f} MB")
    print(f"Chunk size:           {chunk_size} frames")
    print(f"Denoising steps:      {p1.denoising_steps}")
    print(f"{'─'*60}")
    print(f"{'Component':<35} {'Avg ms':>10}")
    print(f"{'─'*60}")

    display_order = [
        ("VAE stream_encode", "vae_stream_encode"),
        ("DiT forward (all 30 blocks)", "dit_forward_total"),
        ("  Blocks 0-9", "dit_blocks_0_to_9"),
        ("  Blocks 10-19", "dit_blocks_10_to_19"),
        ("  Blocks 20-29", "dit_blocks_20_to_29"),
        ("VAE stream_decode", "vae_stream_decode"),
    ]
    for label, key in display_order:
        if key in component_breakdown:
            ms = component_breakdown[key]["avg_ms"]
            print(f"  {label:<33} {ms:>10.2f}")

    print(f"{'─'*60}")
    print(f"  {'Total per chunk':<33} {total_per_chunk_ms:>10.2f}")
    print(f"  {'Estimated FPS':<33} {fps_from_components:>10.2f}")
    print(f"{'='*60}")
    print(f"\nBottleneck: ", end="")

    # Identify bottleneck
    components = {k: component_breakdown.get(k, {}).get("avg_ms", 0) for k in ["vae_stream_encode", "dit_forward_total", "vae_stream_decode"]}
    bottleneck = max(components, key=components.get)
    bottleneck_pct = components[bottleneck] / total_per_chunk_ms * 100 if total_per_chunk_ms > 0 else 0
    print(f"{bottleneck} ({components[bottleneck]:.1f}ms, {bottleneck_pct:.0f}% of total)")

    print(f"\nResults saved: {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Component Profiling")
    parser.add_argument("--config", type=str, default="configs/experiment_configs.yaml")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--checkpoint_folder", type=str, default=None)
    args = parser.parse_args()

    exp_cfg = OmegaConf.load(args.config)
    run_profiling(exp_cfg, video_path=args.video_path, checkpoint_folder=args.checkpoint_folder)


if __name__ == "__main__":
    main()
