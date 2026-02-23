#!/usr/bin/env python3
"""
Phase 4.1: Multi-GPU Pipeline Baseline (4xH100 NVLink)
Run StreamDiffusionV2's InferencePipelineManager across multiple GPUs
and measure throughput fps.

Usage:
    # Must be launched with torchrun:
    torchrun --nproc_per_node=4 scripts/phase4/multi_gpu_baseline.py --config configs/experiment_configs.yaml
    torchrun --nproc_per_node=2 scripts/phase4/multi_gpu_baseline.py --config configs/experiment_configs.yaml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import torch
import torch.distributed as dist
import time
from omegaconf import OmegaConf

from utils.profiler import FPSTracker, get_gpu_memory_info
from scripts.phase1.run_baseline import build_streamdiffv2_config


def get_block_split(world_size, total_blocks=30):
    """Compute balanced block split for N GPUs."""
    blocks_per_gpu = total_blocks // world_size
    remainder = total_blocks % world_size
    splits = []
    start = 0
    for i in range(world_size):
        end = start + blocks_per_gpu + (1 if i < remainder else 0)
        splits.append([start, end])
        start = end
    return splits


def run_multi_gpu(exp_cfg, checkpoint_folder=None, video_path=None):
    """Run multi-GPU pipeline inference."""
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    p1 = exp_cfg.phase1
    prof_cfg = exp_cfg.profiling
    paths = exp_cfg.paths

    sys.path.insert(0, paths.streamdiffv2)
    from streamv2v.inference_pipe import InferencePipelineManager, load_mp4_as_tensor

    cfg = build_streamdiffv2_config(exp_cfg)
    ckpt_folder = checkpoint_folder or os.path.join(paths.checkpoint_folder, "causvid")

    # Compute block split
    total_blocks = 30  # 1.3B model
    block_splits = get_block_split(world_size, total_blocks)
    block_num = torch.tensor(block_splits[rank], dtype=torch.long, device=device)

    if rank == 0:
        print(f"=== Phase 4: Multi-GPU Baseline ===")
        print(f"World size: {world_size}")
        print(f"Block splits: {block_splits}")
        print(f"Resolution: {p1.height}x{p1.width}")
        print()

    # Initialize pipeline manager
    pipeline_mgr = InferencePipelineManager(cfg, device, rank, world_size)
    pipeline_mgr.load_model(ckpt_folder)

    # Prepare input
    chunk_size = p1.chunk_size * cfg.num_frame_per_block
    total_chunks = prof_cfg.num_warmup + prof_cfg.num_iterations
    num_steps = len(cfg.denoising_step_list)

    if video_path and os.path.exists(video_path) and rank == 0:
        input_video = load_mp4_as_tensor(
            video_path,
            max_frames=1 + chunk_size * (total_chunks + num_steps),
            resize_hw=(p1.height, p1.width),
        ).unsqueeze(0).to(dtype=torch.bfloat16, device=device)
    else:
        input_video = None

    prompts = [p1.prompt]

    output_dir = os.path.join(paths.output_root, "phase4", "multi_gpu", f"{world_size}gpu")
    os.makedirs(output_dir, exist_ok=True)

    # Determine block mode
    if rank == 0:
        block_mode = "input"
    elif rank == world_size - 1:
        block_mode = "output"
    else:
        block_mode = "middle"

    # Prepare
    noise = torch.randn(
        1, 1 + cfg.num_frame_per_block, 16,
        p1.height // 16 * 2, p1.width // 16 * 2,
        device=device, dtype=torch.bfloat16
    )

    pipeline_mgr.prepare_pipeline(
        text_prompts=prompts,
        noise=noise,
        block_mode=block_mode,
        current_start=0,
        current_end=pipeline_mgr.pipeline.frame_seq_length * (1 + chunk_size // 4),
        block_num=block_num,
    )

    dist.barrier()

    # ── Run inference ──
    if rank == 0:
        print(f"Running {total_chunks} chunks...")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    noise_scale = p1.noise_scale
    schedule_block = False

    if rank == 0:
        pipeline_mgr.run_rank_0_loop(
            input_video_original=input_video,
            prompts=prompts,
            num_chunks=total_chunks,
            num_steps=num_steps,
            chunk_size=chunk_size,
            block_num=block_num,
            noise_scale=noise_scale,
            schedule_block=schedule_block,
            total_blocks=total_blocks,
        )
    elif rank == world_size - 1:
        pipeline_mgr.run_final_rank_loop(
            prompts=prompts,
            num_chunks=total_chunks,
            num_steps=num_steps,
            chunk_size=chunk_size,
            block_num=block_num,
            noise_scale=noise_scale,
            output_folder=output_dir,
            fps=p1.fps,
            schedule_block=schedule_block,
            total_blocks=total_blocks,
        )
    else:
        pipeline_mgr.run_middle_rank_loop(
            prompts=prompts,
            num_chunks=total_chunks,
            num_steps=num_steps,
            chunk_size=chunk_size,
            block_num=block_num,
            noise_scale=noise_scale,
            schedule_block=schedule_block,
            total_blocks=total_blocks,
        )

    torch.cuda.synchronize()
    dist.barrier()
    total_time = time.perf_counter() - start_time
    total_frames = total_chunks * chunk_size

    # Collect results (rank 0 only)
    if rank == 0:
        results = {
            "phase": "phase4_multi_gpu_baseline",
            "num_gpus": world_size,
            "block_splits": block_splits,
            "resolution": f"{p1.height}x{p1.width}",
            "model": exp_cfg.model.model_type,
            "denoising_steps": p1.denoising_steps,
            "total_frames": total_frames,
            "total_time_s": round(total_time, 2),
            "throughput_fps": round(total_frames / total_time, 2),
            "avg_ms_per_frame": round(total_time / total_frames * 1000, 2),
            "gpu_memory_per_rank": {},
        }

        # Gather memory info from all ranks
        results["gpu_memory_per_rank"][f"rank_{rank}"] = get_gpu_memory_info()

        print(f"\n{'='*60}")
        print(f"MULTI-GPU BASELINE RESULTS ({world_size}xH100)")
        print(f"{'='*60}")
        print(f"Resolution:       {results['resolution']}")
        print(f"Throughput FPS:   {results['throughput_fps']:.2f}")
        print(f"ms/frame:         {results['avg_ms_per_frame']:.2f}")
        print(f"Total time:       {results['total_time_s']:.2f}s")
        print(f"Block splits:     {block_splits}")

        out_path = os.path.join(output_dir, f"multi_gpu_{world_size}_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {out_path}")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_configs.yaml")
    parser.add_argument("--checkpoint_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    args = parser.parse_args()
    exp_cfg = OmegaConf.load(args.config)
    run_multi_gpu(exp_cfg, args.checkpoint_folder, args.video_path)


if __name__ == "__main__":
    main()
