#!/usr/bin/env python3
"""
Phase 2: Run All Optimization Experiments
Runs each experiment sequentially and generates a cumulative effect table.

Usage:
    python scripts/phase2/run_all_optimizations.py --config configs/experiment_configs.yaml
    python scripts/phase2/run_all_optimizations.py --config configs/experiment_configs.yaml --only compile,fp8
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
from omegaconf import OmegaConf


EXPERIMENTS = {
    "compile": ("scripts.phase2.opt_torch_compile", "run_compile_experiment"),
    "stable_fast": ("scripts.phase2.opt_stable_fast", "run_stable_fast_experiment"),
    "fp8": ("scripts.phase2.opt_fp8", "run_fp8_experiment"),
    "attention": ("scripts.phase2.opt_flash_attn", "run_attention_experiment"),
    "denoising_steps": ("scripts.phase2.opt_denoising_steps", "run_step_experiment"),
}


def run_all(exp_cfg, only=None, checkpoint_folder=None):
    """Run selected or all experiments and aggregate results."""
    all_results = {}

    experiments = only.split(",") if only else list(EXPERIMENTS.keys())

    for exp_name in experiments:
        if exp_name not in EXPERIMENTS:
            print(f"WARNING: Unknown experiment '{exp_name}', skipping.")
            continue

        module_path, func_name = EXPERIMENTS[exp_name]
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT: {exp_name}")
        print(f"{'#'*70}")

        try:
            import importlib
            mod = importlib.import_module(module_path)
            func = getattr(mod, func_name)
            result = func(exp_cfg, checkpoint_folder=checkpoint_folder)
            all_results[exp_name] = result
        except Exception as e:
            print(f"ERROR in {exp_name}: {e}")
            all_results[exp_name] = {"error": str(e)}

    # ── Generate cumulative table ──
    print(f"\n{'='*80}")
    print(f"PHASE 2: OPTIMIZATION CUMULATIVE EFFECT TABLE")
    print(f"{'='*80}")
    print(f"{'Configuration':<35} {'FPS':>8} {'ms/frame':>10} {'Quality':>10}")
    print(f"{'─'*65}")

    # Extract key numbers for the table
    table_rows = []

    # Baseline
    if "compile" in all_results and "no_compile" in all_results["compile"]:
        r = all_results["compile"]["no_compile"]
        table_rows.append(("Baseline (BF16, 2-step)", r.get("avg_fps", "?"), r.get("avg_ms_per_frame", "?"), "baseline"))

    # + torch.compile
    if "compile" in all_results and "max-autotune" in all_results["compile"]:
        r = all_results["compile"]["max-autotune"]
        table_rows.append(("+ torch.compile", r.get("avg_fps", "?"), r.get("avg_ms_per_frame", "?"), "same"))

    # + stable-fast
    if "stable_fast" in all_results and "stable_fast" in all_results["stable_fast"]:
        r = all_results["stable_fast"]["stable_fast"]
        table_rows.append(("+ stable-fast", r.get("avg_fps", "?"), r.get("avg_ms_per_frame", "?"), "same"))

    # + FP8
    if "fp8" in all_results:
        for mode in ["fp8_e4m3fn", "fp8_torchao"]:
            if mode in all_results["fp8"] and "error" not in all_results["fp8"][mode]:
                r = all_results["fp8"][mode]
                table_rows.append((f"+ FP8 ({mode})", r.get("avg_fps", "?"), r.get("avg_ms_per_frame", "?"), "measure"))
                break

    # + 1-step
    if "denoising_steps" in all_results and "1-step" in all_results["denoising_steps"]:
        r = all_results["denoising_steps"]["1-step"]
        table_rows.append(("+ 1-step inference", r.get("avg_fps", "?"), r.get("avg_ms_per_frame", "?"), "lower"))

    for label, fps, ms, quality in table_rows:
        fps_str = f"{fps:.2f}" if isinstance(fps, (int, float)) else str(fps)
        ms_str = f"{ms:.2f}" if isinstance(ms, (int, float)) else str(ms)
        print(f"{label:<35} {fps_str:>8} {ms_str:>10} {quality:>10}")

    # Theoretical max
    print(f"{'─'*65}")
    if table_rows:
        best_fps = max((r[1] for r in table_rows if isinstance(r[1], (int, float))), default=0)
        print(f"{'Best measured (single H100)':<35} {best_fps:>8.2f}")

    # Save aggregate
    paths = exp_cfg.paths
    out_path = os.path.join(paths.output_root, "phase2", "all_optimization_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved: {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_configs.yaml")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated list: compile,fp8,...")
    parser.add_argument("--checkpoint_folder", type=str, default=None)
    args = parser.parse_args()
    exp_cfg = OmegaConf.load(args.config)
    run_all(exp_cfg, only=args.only, checkpoint_folder=args.checkpoint_folder)


if __name__ == "__main__":
    main()
