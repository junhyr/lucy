#!/usr/bin/env python3
"""
Phase 4.4: Resolution x GPU Count Matrix
Run the pipeline at different resolutions and GPU counts to fill the matrix.

This script orchestrates multiple runs (single-GPU and multi-GPU) and aggregates results.

Usage:
    # Run the orchestrator (launches sub-processes):
    python scripts/phase4/resolution_gpu_matrix.py --config configs/experiment_configs.yaml

    # Or run individual configs:
    python scripts/phase4/resolution_gpu_matrix.py --config configs/experiment_configs.yaml --resolution 480p --num_gpus 1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import subprocess
import torch
from omegaconf import OmegaConf

from utils.profiler import get_gpu_memory_info


def run_single_config(config_path, resolution, num_gpus, checkpoint_folder=None):
    """Run a single resolution x GPU configuration."""
    if num_gpus == 1:
        cmd = [
            sys.executable, "scripts/phase3/resolution_scaling.py",
            "--config", config_path,
        ]
        if checkpoint_folder:
            cmd.extend(["--checkpoint_folder", checkpoint_folder])
    else:
        cmd = [
            "torchrun", f"--nproc_per_node={num_gpus}",
            "scripts/phase4/multi_gpu_baseline.py",
            "--config", config_path,
        ]
        if checkpoint_folder:
            cmd.extend(["--checkpoint_folder", checkpoint_folder])

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/workspace/lucy-poc")
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
    return result.returncode


def aggregate_results(exp_cfg):
    """Aggregate all phase3 and phase4 results into a matrix."""
    paths = exp_cfg.paths
    p3 = exp_cfg.phase3
    p4 = exp_cfg.phase4

    matrix = {}

    # Load Phase 3 resolution results (1xGPU)
    res_path = os.path.join(paths.output_root, "phase3", "resolution", "resolution_results.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            res_data = json.load(f)
        for res_name, r in res_data.items():
            key = f"1xH100_{res_name}"
            matrix[key] = {
                "resolution": r.get("resolution", res_name),
                "num_gpus": 1,
                "fps": r.get("avg_fps", "?"),
                "ms_per_frame": r.get("avg_ms_per_frame", "?"),
            }

    # Load Phase 4 multi-GPU results
    for ngpu in [2, 4]:
        mgpu_path = os.path.join(paths.output_root, "phase4", "multi_gpu",
                                f"{ngpu}gpu", f"multi_gpu_{ngpu}_results.json")
        if os.path.exists(mgpu_path):
            with open(mgpu_path) as f:
                mgpu_data = json.load(f)
            res = mgpu_data.get("resolution", "480p")
            key = f"{ngpu}xH100_{res}"
            matrix[key] = {
                "resolution": res,
                "num_gpus": ngpu,
                "fps": mgpu_data.get("throughput_fps", "?"),
                "ms_per_frame": mgpu_data.get("avg_ms_per_frame", "?"),
            }

    # ── Print matrix ──
    resolutions = [r["name"] for r in p3.resolutions]
    gpu_configs = [1, 2, 4]

    print(f"\n{'='*70}")
    print(f"RESOLUTION x GPU COUNT FPS MATRIX")
    print(f"{'='*70}")

    header = f"{'GPU':>10}"
    for rn in resolutions:
        header += f"{rn:>12}"
    print(header)
    print("─" * (10 + 12 * len(resolutions)))

    for ngpu in gpu_configs:
        row = f"{ngpu}xH100{'':<5}"
        for rn in resolutions:
            key = f"{ngpu}xH100_{rn}"
            if key in matrix:
                fps = matrix[key]["fps"]
                fps_str = f"{fps:.1f}" if isinstance(fps, (int, float)) else str(fps)
                # Add visual indicator for 30fps
                if isinstance(fps, (int, float)) and fps >= 30:
                    fps_str += " !"
                row += f"{fps_str:>12}"
            else:
                row += f"{'--':>12}"
        print(row)

    print(f"\n('!' = meets 30fps target)")

    # Save
    out_path = os.path.join(paths.output_root, "phase4", "resolution_gpu_matrix.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"\nMatrix saved: {out_path}")

    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_configs.yaml")
    parser.add_argument("--checkpoint_folder", type=str, default=None)
    parser.add_argument("--resolution", type=str, default=None, help="Single resolution to test")
    parser.add_argument("--num_gpus", type=int, default=None, help="Single GPU count to test")
    parser.add_argument("--aggregate_only", action="store_true", help="Only aggregate existing results")
    args = parser.parse_args()

    exp_cfg = OmegaConf.load(args.config)

    if args.aggregate_only:
        aggregate_results(exp_cfg)
        return

    if args.resolution and args.num_gpus:
        run_single_config(args.config, args.resolution, args.num_gpus, args.checkpoint_folder)
    else:
        # Aggregate whatever results exist
        aggregate_results(exp_cfg)


if __name__ == "__main__":
    main()
