#!/usr/bin/env python3
"""
Phase 2.3: FP8 Quantization (H100 Hopper)
Apply float8_e4m3fn to the DiT forward pass while keeping VAE at BF16.
Measure performance and quality impact.

Usage:
    python scripts/phase2/opt_fp8.py --config configs/experiment_configs.yaml
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


def apply_fp8_quantization(model, dtype_str="float8_e4m3fn"):
    """Apply FP8 quantization to DiT linear layers using torch's native FP8 support."""
    dtype_map = {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
    }
    fp8_dtype = dtype_map.get(dtype_str)
    if fp8_dtype is None:
        print(f"WARNING: Unknown FP8 dtype '{dtype_str}', skipping.")
        return model, False, 0

    # Check if current GPU supports FP8
    capability = torch.cuda.get_device_capability()
    if capability[0] < 9:
        print(f"WARNING: FP8 requires SM >= 9.0 (Hopper), got SM {capability[0]}.{capability[1]}. Skipping.")
        return model, False, 0

    quantized_count = 0

    # Apply FP8 to linear layers in transformer blocks
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Skip small layers (embeddings, norms)
            if module.weight.shape[0] < 256 or module.weight.shape[1] < 256:
                continue
            try:
                # Cast weights to FP8 for storage, compute will happen in BF16/FP16
                with torch.no_grad():
                    # Scale factor for FP8
                    w = module.weight.data
                    amax = w.abs().amax()
                    # e4m3fn max representable value is ~448
                    scale = 448.0 / amax.clamp(min=1e-12)
                    w_fp8 = (w * scale).to(fp8_dtype)
                    # Store quantized weight and scale
                    module.register_buffer("weight_fp8", w_fp8)
                    module.register_buffer("weight_scale", torch.tensor(1.0 / scale, device=w.device))
                    quantized_count += 1
            except Exception as e:
                # Skip layers that can't be quantized
                continue

    print(f"Quantized {quantized_count} linear layers to {dtype_str}")
    return model, True, quantized_count


def apply_dynamic_fp8(model):
    """Apply dynamic FP8 quantization using torchao if available."""
    try:
        from torchao.float8 import convert_to_float8_training
        # For inference-only, we use a simpler approach
        from torchao.quantization import quantize_, float8_weight_only
        quantize_(model, float8_weight_only())
        return model, True
    except ImportError:
        print("torchao not available for dynamic FP8. Using manual quantization.")
        return model, False
    except Exception as e:
        print(f"torchao FP8 failed: {e}. Using manual quantization.")
        return model, False


def run_fp8_experiment(exp_cfg, checkpoint_folder=None):
    """Compare BF16 baseline vs FP8 quantized inference."""
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    p1 = exp_cfg.phase1
    fp8_cfg = exp_cfg.phase2.fp8
    paths = exp_cfg.paths
    prof_cfg = exp_cfg.profiling

    sys.path.insert(0, paths.streamdiffv2)
    from streamv2v.inference import SingleGPUInferencePipeline

    cfg = build_streamdiffv2_config(exp_cfg)
    ckpt_folder = checkpoint_folder or os.path.join(paths.checkpoint_folder, "causvid")

    chunk_size = p1.chunk_size * cfg.num_frame_per_block
    total_chunks = prof_cfg.num_warmup + prof_cfg.num_iterations
    num_steps = len(cfg.denoising_step_list)
    prompts = [p1.prompt]

    results = {}

    for mode in ["bf16_baseline", "fp8_e4m3fn", "fp8_torchao"]:
        print(f"\n{'='*60}")
        print(f"Testing: {mode}")
        print(f"{'='*60}")

        pipeline_mgr = SingleGPUInferencePipeline(cfg, device)
        pipeline_mgr.load_model(ckpt_folder)

        if mode == "fp8_e4m3fn":
            dit_model = pipeline_mgr.pipeline.generator.model
            dit_model, success, count = apply_fp8_quantization(dit_model, fp8_cfg.dit_dtype)
            if not success:
                del pipeline_mgr
                torch.cuda.empty_cache()
                continue

        elif mode == "fp8_torchao":
            dit_model = pipeline_mgr.pipeline.generator.model
            dit_model, success = apply_dynamic_fp8(dit_model)
            if not success:
                del pipeline_mgr
                torch.cuda.empty_cache()
                continue

        output_dir = os.path.join(paths.output_root, "phase2", "fp8", mode)
        os.makedirs(output_dir, exist_ok=True)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()

        try:
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
        except Exception as e:
            print(f"  ERROR: {e}")
            results[mode] = {"error": str(e)}
            del pipeline_mgr
            torch.cuda.empty_cache()
            continue

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
        print(f"  Peak mem: {results[mode]['gpu_memory']['max_allocated_mb']:.0f}MB")

        del pipeline_mgr
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"FP8 QUANTIZATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Mode':<20} {'FPS':>8} {'ms/frame':>10} {'Peak MB':>10}")
    print(f"{'─'*50}")
    baseline_fps = results.get("bf16_baseline", {}).get("avg_fps", 1)
    for mode, r in results.items():
        if "error" in r:
            print(f"{mode:<20} {'ERROR':>8}")
            continue
        speedup = r["avg_fps"] / baseline_fps if baseline_fps > 0 else 0
        peak_mb = r["gpu_memory"]["max_allocated_mb"]
        print(f"{mode:<20} {r['avg_fps']:>8.2f} {r['avg_ms_per_frame']:>10.2f} {peak_mb:>10.0f}")

    out_path = os.path.join(paths.output_root, "phase2", "fp8", "fp8_results.json")
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
    run_fp8_experiment(exp_cfg, args.checkpoint_folder)


if __name__ == "__main__":
    main()
