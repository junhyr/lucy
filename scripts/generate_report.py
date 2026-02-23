#!/usr/bin/env python3
"""
Generate Final Report
Aggregates all phase results into the final decision tables.

Usage:
    python scripts/generate_report.py --config configs/experiment_configs.yaml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import glob
from datetime import datetime
from omegaconf import OmegaConf


def load_json_safe(path):
    """Load JSON file or return empty dict."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def generate_report(exp_cfg):
    """Generate comprehensive report from all experiment results."""
    paths = exp_cfg.paths
    output_root = paths.output_root

    report_lines = []
    report_data = {"generated_at": datetime.now().isoformat(), "phases": {}}

    def section(title):
        report_lines.append(f"\n{'='*70}")
        report_lines.append(title)
        report_lines.append(f"{'='*70}")

    def line(text=""):
        report_lines.append(text)

    section("LUCY 2.0 CLONE — INFERENCE PoC RESULTS")
    line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    line(f"Model: {exp_cfg.model.model_type}")
    line()

    # ── Phase 1: Baseline ──
    section("PHASE 1: BASELINE")
    baseline = load_json_safe(os.path.join(output_root, "phase1", "baseline", "baseline_results.json"))
    profiling = load_json_safe(os.path.join(output_root, "phase1", "profiling", "profiling_results.json"))

    if baseline:
        line(f"Resolution:     {baseline.get('resolution', '?')}")
        line(f"Steps:          {baseline.get('denoising_steps', '?')}")
        line(f"Baseline FPS:   {baseline.get('avg_fps', '?')}")
        line(f"ms/frame:       {baseline.get('avg_ms_per_frame', '?')}")
        report_data["phases"]["phase1"] = baseline

    if profiling:
        line()
        line("Component Breakdown:")
        breakdown = profiling.get("component_breakdown", {})
        for comp, data in breakdown.items():
            line(f"  {comp:<35} {data.get('avg_ms', '?'):>8} ms")
        line(f"  {'Total per chunk':<35} {profiling.get('total_per_chunk_ms', '?'):>8} ms")
        report_data["phases"]["phase1_profiling"] = profiling

    # ── Phase 2: Optimizations ──
    section("PHASE 2: OPTIMIZATION CUMULATIVE EFFECT")
    all_opt = load_json_safe(os.path.join(output_root, "phase2", "all_optimization_results.json"))

    if all_opt:
        line(f"{'Configuration':<35} {'FPS':>8} {'ms/frame':>10} {'Quality':>10}")
        line(f"{'─'*65}")

        # Compile results
        compile_data = all_opt.get("compile", {})
        if "no_compile" in compile_data:
            r = compile_data["no_compile"]
            line(f"{'Baseline (BF16, 2-step)':<35} {r.get('avg_fps', '?'):>8} {r.get('avg_ms_per_frame', '?'):>10} {'baseline':>10}")
        if "max-autotune" in compile_data:
            r = compile_data["max-autotune"]
            line(f"{'+ torch.compile':<35} {r.get('avg_fps', '?'):>8} {r.get('avg_ms_per_frame', '?'):>10} {'same':>10}")

        # Stable-fast
        sf_data = all_opt.get("stable_fast", {})
        if "stable_fast" in sf_data:
            r = sf_data["stable_fast"]
            line(f"{'+ stable-fast':<35} {r.get('avg_fps', '?'):>8} {r.get('avg_ms_per_frame', '?'):>10} {'same':>10}")

        # FP8
        fp8_data = all_opt.get("fp8", {})
        for mode in ["fp8_e4m3fn", "fp8_torchao"]:
            if mode in fp8_data and "error" not in fp8_data[mode]:
                r = fp8_data[mode]
                line(f"{'+ FP8 (' + mode + ')':<35} {r.get('avg_fps', '?'):>8} {r.get('avg_ms_per_frame', '?'):>10} {'measure':>10}")
                break

        # Steps
        step_data = all_opt.get("denoising_steps", {})
        if "1-step" in step_data and "error" not in step_data.get("1-step", {}):
            r = step_data["1-step"]
            line(f"{'+ 1-step inference':<35} {r.get('avg_fps', '?'):>8} {r.get('avg_ms_per_frame', '?'):>10} {'lower':>10}")

        report_data["phases"]["phase2"] = all_opt
    else:
        # Try loading individual results
        for exp in ["compile", "stable_fast", "fp8", "attention", "denoising_steps"]:
            pattern = os.path.join(output_root, "phase2", exp, f"*_results.json")
            for f in glob.glob(pattern):
                data = load_json_safe(f)
                line(f"  {exp}: {json.dumps({k: v.get('avg_fps') for k, v in data.items() if isinstance(v, dict) and 'avg_fps' in v}, indent=None)}")

    # ── Phase 3: Resolution Scaling ──
    section("PHASE 3: RESOLUTION x MODEL SIZE")
    res_data = load_json_safe(os.path.join(output_root, "phase3", "resolution", "resolution_results.json"))
    model_data = load_json_safe(os.path.join(output_root, "phase3", "model_size", "model_size_results.json"))
    vae_data = load_json_safe(os.path.join(output_root, "phase3", "vae_bottleneck", "vae_results.json"))

    if res_data:
        line("Resolution scaling (1xH100, measured):")
        line(f"{'Resolution':<12} {'FPS':>8} {'ms/frame':>10} {'Peak MB':>10}")
        line(f"{'─'*45}")
        for name, r in res_data.items():
            if "error" in r:
                line(f"{r.get('resolution', name):<12} {'OOM':>8}")
            else:
                line(f"{r.get('resolution', name):<12} {r.get('avg_fps', '?'):>8} {r.get('avg_ms_per_frame', '?'):>10} {r.get('peak_memory_mb', '?'):>10}")

    if model_data and "per_block_times" in model_data:
        line()
        line("Model size simulation (estimated FPS):")
        models = list(model_data["per_block_times"].keys())
        header = f"{'Resolution':<12}" + "".join(f"{m:>12}" for m in models)
        line(header)
        line("─" * (12 + 12 * len(models)))

        # Get all resolutions from first model
        first_model = model_data["per_block_times"].get(models[0], {})
        for rn in first_model:
            row = f"{rn:<12}"
            for mn in models:
                bt = model_data["per_block_times"].get(mn, {}).get(rn, {})
                if "error" in bt:
                    row += f"{'OOM':>12}"
                elif "estimated_fps" in bt:
                    row += f"{bt['estimated_fps']:>11.1f}f"
                else:
                    row += f"{'?':>12}"
            line(row)

    if vae_data:
        line()
        line("VAE bottleneck analysis:")
        for name, r in vae_data.items():
            if "error" not in r:
                line(f"  {r.get('resolution', name)}: encode={r.get('encode_avg_ms', '?')}ms, "
                     f"decode={r.get('decode_avg_ms', '?')}ms, total={r.get('total_vae_ms', '?')}ms")

    report_data["phases"]["phase3"] = {
        "resolution": res_data, "model_size": model_data, "vae": vae_data
    }

    # ── Phase 4: Multi-GPU ──
    section("PHASE 4: MULTI-GPU PIPELINE")
    matrix = load_json_safe(os.path.join(output_root, "phase4", "resolution_gpu_matrix.json"))
    comm = load_json_safe(os.path.join(output_root, "phase4", "comm_profile", "comm_profile_4gpu.json"))

    if comm:
        line("Communication profiling (NVLink):")
        p2p = comm.get("p2p_transfer", {})
        for name, data in p2p.items():
            if isinstance(data, dict) and "avg_ms" in data:
                line(f"  {name}: {data['avg_ms']:.4f}ms (BW: {data.get('effective_bandwidth_gbps', '?')} GB/s)")

        pipe = comm.get("pipeline_pass", {})
        if pipe:
            line(f"  Pipeline pass ({pipe.get('num_hops', '?')} hops): {pipe.get('avg_total_ms', '?')}ms")

    if matrix:
        line()
        line("Resolution x GPU matrix:")
        for key, data in matrix.items():
            fps = data.get("fps", "?")
            marker = " <-- 30fps!" if isinstance(fps, (int, float)) and fps >= 30 else ""
            line(f"  {key}: {fps} fps{marker}")

    report_data["phases"]["phase4"] = {"matrix": matrix, "comm": comm}

    # ── Final Conclusion ──
    section("FINAL CONCLUSION TABLE")
    line(f"{'Model':<7} {'Res':<6} {'GPU':<8} {'Steps':>5} {'Optimiz':<18} {'Throughput':>10} {'Latency':>10} {'30fps?':>7}")
    line(f"{'─'*75}")

    # This will be filled with actual data when experiments are run
    conclusion_rows = [
        ("1.3B", "480p", "1xH100", "2", "compile+fp8", "?", "?", "?"),
        ("1.3B", "480p", "4xH100", "2", "pipeline+fp8", "?", "?", "?"),
        ("1.3B", "720p", "4xH100", "2", "pipeline+fp8", "?", "?", "?"),
        ("1.3B", "720p", "4xH100", "1", "pipeline+fp8", "?", "?", "?"),
        ("5B*", "720p", "4xH100", "2", "pipeline+fp8", "?(est)", "?(est)", "?"),
        ("5B*", "720p", "8xH100", "1", "pipeline+fp8", "?(est)", "?(est)", "?"),
    ]

    # Try to fill in from actual data
    for row in conclusion_rows:
        line(f"{row[0]:<7} {row[1]:<6} {row[2]:<8} {row[3]:>5} {row[4]:<18} {row[5]:>10} {row[6]:>10} {row[7]:>7}")

    line()
    line("RECOMMENDATION:")
    # Determine recommendation based on best measured fps
    best_fps = 0
    for phase_data in [res_data, matrix]:
        if isinstance(phase_data, dict):
            for v in phase_data.values():
                if isinstance(v, dict):
                    fps = v.get("avg_fps") or v.get("fps") or v.get("throughput_fps") or 0
                    if isinstance(fps, (int, float)):
                        best_fps = max(best_fps, fps)

    if best_fps >= 30:
        line(f"  Best measured: {best_fps:.1f} fps — 30fps ACHIEVABLE")
        line("  -> Proceed with V2V training + Self-Forcing")
    elif best_fps >= 20:
        line(f"  Best measured: {best_fps:.1f} fps — CLOSE to 30fps")
        line("  -> Invest in custom Triton kernels or resolution compromise")
    elif best_fps > 0:
        line(f"  Best measured: {best_fps:.1f} fps — BELOW target")
        line("  -> Need fundamental architecture change (smaller model, aggressive distillation)")
    else:
        line("  No fps data found — run experiments first!")

    # ── Save report ──
    report_text = "\n".join(report_lines)

    out_dir = os.path.join(output_root, "report")
    os.makedirs(out_dir, exist_ok=True)

    # Text report
    text_path = os.path.join(out_dir, "final_report.txt")
    with open(text_path, "w") as f:
        f.write(report_text)

    # JSON data
    json_path = os.path.join(out_dir, "final_report_data.json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    print(report_text)
    print(f"\n\nReport saved: {text_path}")
    print(f"Data saved:   {json_path}")

    return report_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_configs.yaml")
    args = parser.parse_args()
    exp_cfg = OmegaConf.load(args.config)
    generate_report(exp_cfg)


if __name__ == "__main__":
    main()
