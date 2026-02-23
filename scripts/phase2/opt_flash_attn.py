#!/usr/bin/env python3
"""
Phase 2.4: flash_attn vs flex_attention Comparison
Compare attention backends for causal inference with KV cache.

Current code uses flex_attention (compiled, max-autotune).
Alternative: flash_attn (flash_attn_interface).
Check: causal attention + KV cache + sink token compatibility.

Usage:
    python scripts/phase2/opt_flash_attn.py --config configs/experiment_configs.yaml
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


def patch_attention_backend(pipeline, backend="flash_attn"):
    """
    Patch the attention implementation in CausalWanSelfAttention.

    The default code uses:
      - flex_attention (compiled, max-autotune) for prepare() without KV cache
      - flash_attn_interface for inference_stream() with KV cache

    We test making everything use one backend consistently.
    """
    model = pipeline.generator.model

    if not hasattr(model, 'blocks'):
        print("WARNING: Model doesn't have 'blocks' attribute, cannot patch attention.")
        return False

    if backend == "flash_attn":
        try:
            from flash_attn import flash_attn_interface
            print("Using flash_attn backend")
        except ImportError:
            print("ERROR: flash_attn not installed")
            return False

    elif backend == "xformers":
        try:
            from xformers.ops import memory_efficient_attention
            print("Using xformers backend")

            # Patch each block's self_attn to use xformers
            for block in model.blocks:
                attn = block.self_attn
                original_forward = attn.forward

                def make_xformers_forward(orig_fwd, attn_module):
                    def xformers_forward(x, seq_lens, grid_sizes, freqs, block_mask,
                                        kv_cache=None, current_start=0, current_end=0):
                        b, s, n, d = *x.shape[:2], attn_module.num_heads, attn_module.head_dim

                        q = attn_module.norm_q(attn_module.q(x)).unflatten(2, (n, d))
                        k = attn_module.norm_k(attn_module.k(x)).unflatten(2, (n, d))
                        v = attn_module.v(x).unflatten(2, (n, d))

                        if kv_cache is not None:
                            # Use cached K, V for efficient inference
                            from causvid.models.wan.causal_model import causal_rope_apply
                            roped_q = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start)
                            roped_k = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start)

                            kv_cache["k"][:, current_start:current_end] = roped_k
                            kv_cache["v"][:, current_start:current_end] = v

                            # xformers expects (B, S, H, D)
                            out = memory_efficient_attention(
                                roped_q,
                                kv_cache["k"][:, :current_end],
                                kv_cache["v"][:, :current_end],
                            )
                        else:
                            return orig_fwd(x, seq_lens, grid_sizes, freqs, block_mask,
                                          kv_cache, current_start, current_end)

                        out = out.flatten(2)
                        out = attn_module.o(out)
                        return out

                    return xformers_forward

                attn.forward = make_xformers_forward(original_forward, attn)

        except ImportError:
            print("ERROR: xformers not installed")
            return False

    elif backend == "flex_attention":
        # Default — no patching needed
        print("Using flex_attention backend (default)")

    return True


def run_attention_experiment(exp_cfg, checkpoint_folder=None):
    """Compare attention backends."""
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    p1 = exp_cfg.phase1
    paths = exp_cfg.paths
    prof_cfg = exp_cfg.profiling
    attn_cfg = exp_cfg.phase2.attention

    sys.path.insert(0, paths.streamdiffv2)
    from streamv2v.inference import SingleGPUInferencePipeline

    cfg = build_streamdiffv2_config(exp_cfg)
    ckpt_folder = checkpoint_folder or os.path.join(paths.checkpoint_folder, "causvid")

    chunk_size = p1.chunk_size * cfg.num_frame_per_block
    total_chunks = prof_cfg.num_warmup + prof_cfg.num_iterations
    num_steps = len(cfg.denoising_step_list)
    prompts = [p1.prompt]

    results = {}

    for backend in attn_cfg.variants:
        print(f"\n{'='*60}")
        print(f"Testing attention backend: {backend}")
        print(f"{'='*60}")

        pipeline_mgr = SingleGPUInferencePipeline(cfg, device)
        pipeline_mgr.load_model(ckpt_folder)

        if backend != "flex_attention":
            success = patch_attention_backend(pipeline_mgr.pipeline, backend)
            if not success:
                results[backend] = {"error": f"Failed to apply {backend}"}
                del pipeline_mgr
                torch.cuda.empty_cache()
                continue

        output_dir = os.path.join(paths.output_root, "phase2", "attention", backend)
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
            results[backend] = {"error": str(e)}
            del pipeline_mgr
            torch.cuda.empty_cache()
            continue

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        total_frames = total_chunks * chunk_size

        results[backend] = {
            "total_frames": total_frames,
            "total_time_s": round(total_time, 2),
            "avg_fps": round(total_frames / total_time, 2),
            "avg_ms_per_frame": round(total_time / total_frames * 1000, 2),
            "gpu_memory": get_gpu_memory_info(),
        }

        print(f"  FPS: {results[backend]['avg_fps']:.2f}")

        del pipeline_mgr
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"ATTENTION BACKEND COMPARISON")
    print(f"{'='*60}")
    print(f"{'Backend':<20} {'FPS':>8} {'ms/frame':>10} {'Peak MB':>10}")
    print(f"{'─'*50}")
    for backend, r in results.items():
        if "error" in r:
            print(f"{backend:<20} ERROR: {r['error']}")
        else:
            print(f"{backend:<20} {r['avg_fps']:>8.2f} {r['avg_ms_per_frame']:>10.2f} {r['gpu_memory']['max_allocated_mb']:>10.0f}")

    out_path = os.path.join(paths.output_root, "phase2", "attention", "attention_results.json")
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
    run_attention_experiment(exp_cfg, args.checkpoint_folder)


if __name__ == "__main__":
    main()
