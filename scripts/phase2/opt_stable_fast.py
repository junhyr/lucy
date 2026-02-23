#!/usr/bin/env python3
"""
Phase 2.2: stable-fast Optimization
Apply stable-fast compiler optimizations (CUDNN fusion, JIT tracing, etc.)

Usage:
    python scripts/phase2/opt_stable_fast.py --config configs/experiment_configs.yaml
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


def apply_stable_fast_optimizations(pipeline, sf_cfg):
    """Apply stable-fast optimizations to the pipeline."""
    try:
        from sfast.compilers.diffusion_pipeline_compiler import (
            compile as sfast_compile,
            CompilationConfig,
        )
    except ImportError:
        print("WARNING: stable-fast not installed. Trying alternative import...")
        try:
            sys.path.insert(0, "/workspace/repos/stable-fast")
            from sfast.compilers.diffusion_pipeline_compiler import (
                compile as sfast_compile,
                CompilationConfig,
            )
        except ImportError:
            print("ERROR: Cannot import stable-fast. Skipping.")
            return pipeline, False

    config = CompilationConfig.Default()
    config.enable_jit = sf_cfg.enable_jit
    config.enable_jit_freeze = sf_cfg.enable_jit_freeze
    config.enable_cnn_optimization = sf_cfg.enable_cnn_optimization
    config.enable_fused_linear_geglu = sf_cfg.enable_fused_linear_geglu
    config.prefer_lowp_gemm = sf_cfg.prefer_lowp_gemm
    config.enable_cuda_graph = sf_cfg.enable_cuda_graph
    config.enable_triton = sf_cfg.enable_triton

    print(f"stable-fast config:")
    print(f"  JIT: {config.enable_jit}, JIT freeze: {config.enable_jit_freeze}")
    print(f"  CNN opt: {config.enable_cnn_optimization}")
    print(f"  Fused GEGLU: {config.enable_fused_linear_geglu}")
    print(f"  Low-precision GEMM: {config.prefer_lowp_gemm}")
    print(f"  CUDA graph: {config.enable_cuda_graph}")
    print(f"  Triton: {config.enable_triton}")

    compiled = sfast_compile(pipeline, config)
    return compiled, True


def run_stable_fast_experiment(exp_cfg, checkpoint_folder=None):
    """Compare baseline vs stable-fast optimized inference."""
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    p1 = exp_cfg.phase1
    sf_cfg = exp_cfg.phase2.stable_fast
    paths = exp_cfg.paths
    prof_cfg = exp_cfg.profiling

    sys.path.insert(0, os.path.join(paths.streamdiffv2, ".."))
    from StreamDiffusionV2.streamv2v.inference import SingleGPUInferencePipeline

    cfg = build_streamdiffv2_config(exp_cfg)
    ckpt_folder = checkpoint_folder or os.path.join(paths.checkpoint_folder, "causvid")

    chunk_size = p1.chunk_size * cfg.num_frame_per_block
    total_chunks = prof_cfg.num_warmup + prof_cfg.num_iterations
    num_steps = len(cfg.denoising_step_list)
    prompts = [p1.prompt]

    results = {}

    for mode in ["baseline", "stable_fast"]:
        print(f"\n{'='*60}")
        print(f"Testing: {mode}")
        print(f"{'='*60}")

        pipeline_mgr = SingleGPUInferencePipeline(cfg, device)
        pipeline_mgr.load_model(ckpt_folder)

        if mode == "stable_fast":
            print("Applying stable-fast optimizations...")
            compile_start = time.perf_counter()
            pipeline_mgr.pipeline, success = apply_stable_fast_optimizations(
                pipeline_mgr.pipeline, sf_cfg
            )
            compile_time = time.perf_counter() - compile_start
            if not success:
                print("stable-fast failed, skipping.")
                del pipeline_mgr
                torch.cuda.empty_cache()
                continue
            print(f"Compilation time: {compile_time:.1f}s")
            results[mode + "_compile_time_s"] = round(compile_time, 1)

        output_dir = os.path.join(paths.output_root, "phase2", "stable_fast", mode)
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

        results[mode] = {
            "total_frames": total_frames,
            "total_time_s": round(total_time, 2),
            "avg_fps": round(total_frames / total_time, 2),
            "avg_ms_per_frame": round(total_time / total_frames * 1000, 2),
            "gpu_memory": get_gpu_memory_info(),
        }

        print(f"  FPS: {results[mode]['avg_fps']:.2f}")

        del pipeline_mgr
        torch.cuda.empty_cache()

    # Summary
    if "baseline" in results and "stable_fast" in results:
        speedup = results["stable_fast"]["avg_fps"] / results["baseline"]["avg_fps"]
        print(f"\n{'='*60}")
        print(f"STABLE-FAST RESULTS")
        print(f"{'='*60}")
        print(f"Baseline FPS:     {results['baseline']['avg_fps']:.2f}")
        print(f"stable-fast FPS:  {results['stable_fast']['avg_fps']:.2f}")
        print(f"Speedup:          {speedup:.2f}x")
        results["speedup"] = round(speedup, 2)

    out_path = os.path.join(paths.output_root, "phase2", "stable_fast", "stable_fast_results.json")
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
    run_stable_fast_experiment(exp_cfg, args.checkpoint_folder)


if __name__ == "__main__":
    main()
