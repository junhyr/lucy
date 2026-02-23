#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Lucy 2.0 PoC — Phase Runner
# Run individual phases or all phases sequentially.
#
# Usage:
#   bash run_phase.sh setup              # Environment setup
#   bash run_phase.sh 1                  # Phase 1: Baseline
#   bash run_phase.sh 2                  # Phase 2: Optimizations (all)
#   bash run_phase.sh 2 compile          # Phase 2: Just torch.compile
#   bash run_phase.sh 3                  # Phase 3: Resolution scaling
#   bash run_phase.sh 4                  # Phase 4: Multi-GPU (requires 4xH100)
#   bash run_phase.sh report             # Generate final report
#   bash run_phase.sh all                # Run everything
# ──────────────────────────────────────────────────────────────
set -euo pipefail

CONFIG="configs/experiment_configs.yaml"
CKPT="${CHECKPOINT_FOLDER:-/workspace/checkpoints/causvid}"
VIDEO="${VIDEO_PATH:-}"

phase="${1:-help}"
sub="${2:-}"

case "$phase" in
  setup)
    echo "=== Setting up environment ==="
    bash scripts/setup/runpod_setup.sh
    ;;

  1|phase1)
    echo "=== Phase 1: Baseline + Profiling ==="
    python scripts/phase1/run_baseline.py --config "$CONFIG" --checkpoint_folder "$CKPT" ${VIDEO:+--video_path "$VIDEO"}
    echo ""
    python scripts/phase1/profile_components.py --config "$CONFIG" --checkpoint_folder "$CKPT" ${VIDEO:+--video_path "$VIDEO"}
    ;;

  2|phase2)
    echo "=== Phase 2: Single-GPU Optimizations ==="
    if [ -n "$sub" ]; then
      case "$sub" in
        compile)
          python scripts/phase2/opt_torch_compile.py --config "$CONFIG" --checkpoint_folder "$CKPT"
          ;;
        stable_fast|sfast)
          python scripts/phase2/opt_stable_fast.py --config "$CONFIG" --checkpoint_folder "$CKPT"
          ;;
        fp8)
          python scripts/phase2/opt_fp8.py --config "$CONFIG" --checkpoint_folder "$CKPT"
          ;;
        attention|attn)
          python scripts/phase2/opt_flash_attn.py --config "$CONFIG" --checkpoint_folder "$CKPT"
          ;;
        steps|denoising)
          python scripts/phase2/opt_denoising_steps.py --config "$CONFIG" --checkpoint_folder "$CKPT" ${VIDEO:+--video_path "$VIDEO"}
          ;;
        all)
          python scripts/phase2/run_all_optimizations.py --config "$CONFIG" --checkpoint_folder "$CKPT"
          ;;
        *)
          echo "Unknown phase2 sub-experiment: $sub"
          echo "Options: compile, stable_fast, fp8, attention, steps, all"
          exit 1
          ;;
      esac
    else
      python scripts/phase2/run_all_optimizations.py --config "$CONFIG" --checkpoint_folder "$CKPT"
    fi
    ;;

  3|phase3)
    echo "=== Phase 3: Resolution Scaling ==="
    if [ -n "$sub" ]; then
      case "$sub" in
        resolution)
          python scripts/phase3/resolution_scaling.py --config "$CONFIG" --checkpoint_folder "$CKPT"
          ;;
        vae)
          python scripts/phase3/vae_bottleneck.py --config "$CONFIG" --checkpoint_folder "$CKPT"
          ;;
        model_size|sim)
          python scripts/phase3/model_size_sim.py --config "$CONFIG" --checkpoint_folder "$CKPT"
          ;;
        *)
          echo "Unknown phase3 sub-experiment: $sub"
          echo "Options: resolution, vae, model_size"
          exit 1
          ;;
      esac
    else
      python scripts/phase3/resolution_scaling.py --config "$CONFIG" --checkpoint_folder "$CKPT"
      python scripts/phase3/vae_bottleneck.py --config "$CONFIG" --checkpoint_folder "$CKPT"
      python scripts/phase3/model_size_sim.py --config "$CONFIG" --checkpoint_folder "$CKPT"
    fi
    ;;

  4|phase4)
    echo "=== Phase 4: Multi-GPU Pipeline ==="
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
    echo "Detected GPUs: $NUM_GPUS"

    if [ "$NUM_GPUS" -lt 2 ]; then
      echo "ERROR: Phase 4 requires at least 2 GPUs. Only $NUM_GPUS detected."
      echo "Running single-GPU throughput/latency instead..."
      python scripts/phase4/throughput_latency.py --config "$CONFIG" --checkpoint_folder "$CKPT" --num_gpus 1
      exit 0
    fi

    if [ -n "$sub" ]; then
      case "$sub" in
        baseline)
          torchrun --nproc_per_node="$NUM_GPUS" scripts/phase4/multi_gpu_baseline.py --config "$CONFIG" --checkpoint_folder "$CKPT"
          ;;
        comm)
          torchrun --nproc_per_node="$NUM_GPUS" scripts/phase4/comm_profile.py --config "$CONFIG"
          ;;
        throughput)
          python scripts/phase4/throughput_latency.py --config "$CONFIG" --checkpoint_folder "$CKPT" --num_gpus 1
          ;;
        matrix)
          python scripts/phase4/resolution_gpu_matrix.py --config "$CONFIG" --checkpoint_folder "$CKPT" --aggregate_only
          ;;
        *)
          echo "Unknown phase4 sub-experiment: $sub"
          echo "Options: baseline, comm, throughput, matrix"
          exit 1
          ;;
      esac
    else
      torchrun --nproc_per_node="$NUM_GPUS" scripts/phase4/comm_profile.py --config "$CONFIG"
      torchrun --nproc_per_node="$NUM_GPUS" scripts/phase4/multi_gpu_baseline.py --config "$CONFIG" --checkpoint_folder "$CKPT"
      python scripts/phase4/throughput_latency.py --config "$CONFIG" --checkpoint_folder "$CKPT" --num_gpus 1
      python scripts/phase4/resolution_gpu_matrix.py --config "$CONFIG" --aggregate_only
    fi
    ;;

  report)
    echo "=== Generating Final Report ==="
    python scripts/generate_report.py --config "$CONFIG"
    ;;

  all)
    echo "=== Running ALL Phases ==="
    bash "$0" 1
    bash "$0" 2
    bash "$0" 3
    # Phase 4 only if multi-GPU available
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
    if [ "$NUM_GPUS" -ge 2 ]; then
      bash "$0" 4
    else
      echo "Skipping Phase 4 (requires multi-GPU)"
      python scripts/phase4/throughput_latency.py --config "$CONFIG" --checkpoint_folder "$CKPT" --num_gpus 1
    fi
    bash "$0" report
    ;;

  help|*)
    echo "Lucy 2.0 PoC — Phase Runner"
    echo ""
    echo "Usage: bash run_phase.sh <phase> [sub-experiment]"
    echo ""
    echo "Phases:"
    echo "  setup          Environment setup (RunPod)"
    echo "  1 | phase1     Baseline + component profiling"
    echo "  2 | phase2     Single-GPU optimizations"
    echo "                   Sub: compile, stable_fast, fp8, attention, steps, all"
    echo "  3 | phase3     Resolution scaling"
    echo "                   Sub: resolution, vae, model_size"
    echo "  4 | phase4     Multi-GPU pipeline (requires 2+ GPUs)"
    echo "                   Sub: baseline, comm, throughput, matrix"
    echo "  report         Generate final report"
    echo "  all            Run everything"
    echo ""
    echo "Environment variables:"
    echo "  CHECKPOINT_FOLDER  Path to model checkpoint (default: /workspace/checkpoints/causvid)"
    echo "  VIDEO_PATH         Path to input video (optional, for V2V mode)"
    ;;
esac
