#!/usr/bin/env python3
"""
Phase 4.2: GPU Inter-Communication Profiling
Measure P2P tensor transfer times over NVLink.
Test various tensor sizes to understand communication overhead.

Usage:
    torchrun --nproc_per_node=4 scripts/phase4/comm_profile.py --config configs/experiment_configs.yaml
    torchrun --nproc_per_node=2 scripts/phase4/comm_profile.py --config configs/experiment_configs.yaml
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


def measure_p2p_transfer(src_rank, dst_rank, tensor, num_warmup=5, num_iters=50):
    """Measure P2P transfer time between two ranks."""
    rank = dist.get_rank()

    # Warmup
    for _ in range(num_warmup):
        if rank == src_rank:
            dist.send(tensor, dst=dst_rank)
        elif rank == dst_rank:
            dist.recv(tensor, src=src_rank)
        dist.barrier()

    # Measure
    times = []
    for _ in range(num_iters):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()

        if rank == src_rank:
            dist.send(tensor, dst=dst_rank)
        elif rank == dst_rank:
            dist.recv(tensor, src=src_rank)

        torch.cuda.synchronize()
        dist.barrier()
        elapsed = (time.perf_counter() - start) * 1000  # ms

        if rank == src_rank:
            times.append(elapsed)

    if rank == src_rank:
        return {
            "avg_ms": round(sum(times) / len(times), 4),
            "min_ms": round(min(times), 4),
            "max_ms": round(max(times), 4),
        }
    return None


def run_comm_profile(exp_cfg):
    """Profile GPU communication."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    p4_comm = exp_cfg.phase4.communication
    paths = exp_cfg.paths

    if rank == 0:
        print(f"=== Phase 4.2: Communication Profiling ===")
        print(f"World size: {world_size}")
        print()

    results = {}

    # ── Test 1: P2P transfer at various sizes ──
    if p4_comm.measure_p2p:
        size_results = {}

        # Realistic activation tensor sizes
        test_sizes = {
            "1MB": 1 * 1024 * 1024 // 2,    # BF16 elements for 1MB
            "4.8MB_activation": None,         # Will compute based on model dims
            "10MB": 10 * 1024 * 1024 // 2,
            "50MB": 50 * 1024 * 1024 // 2,
        }

        # Compute actual activation size: [B=1, seq_len=1560, dim=1536] at BF16
        seq_len = (exp_cfg.phase1.height // 16) * (exp_cfg.phase1.width // 16)
        dim = exp_cfg.model.dim
        activation_elements = 1 * seq_len * dim
        activation_mb = activation_elements * 2 / (1024 * 1024)
        test_sizes["4.8MB_activation"] = activation_elements

        if rank == 0:
            print(f"Activation tensor: [1, {seq_len}, {dim}] = {activation_mb:.1f} MB")
            print(f"NVLink theoretical: 900 GB/s → {activation_mb / 900:.4f} ms\n")

        for size_name, num_elements in test_sizes.items():
            tensor = torch.randn(num_elements, device=device, dtype=torch.bfloat16)
            size_mb = num_elements * 2 / (1024 * 1024)

            if rank == 0:
                print(f"Testing: {size_name} ({size_mb:.1f} MB)...")

            # Test rank 0 → rank 1
            result = measure_p2p_transfer(0, 1, tensor, num_warmup=5, num_iters=50)

            if rank == 0 and result:
                # Calculate effective bandwidth
                bw_gbps = (size_mb / 1024) / (result["avg_ms"] / 1000)
                result["effective_bandwidth_gbps"] = round(bw_gbps, 1)
                result["size_mb"] = round(size_mb, 1)
                size_results[size_name] = result
                print(f"  Avg: {result['avg_ms']:.4f} ms, BW: {bw_gbps:.1f} GB/s")

            del tensor
            torch.cuda.empty_cache()

        results["p2p_transfer"] = size_results

    # ── Test 2: All-reduce latency ──
    if rank == 0:
        print(f"\nTesting all-reduce...")

    allreduce_results = {}
    for size_mb_target in [1, 10, 50]:
        num_elements = size_mb_target * 1024 * 1024 // 2
        tensor = torch.randn(num_elements, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(50):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        if rank == 0:
            avg_ms = sum(times) / len(times)
            allreduce_results[f"{size_mb_target}MB"] = {
                "avg_ms": round(avg_ms, 4),
                "min_ms": round(min(times), 4),
            }
            print(f"  All-reduce {size_mb_target}MB: {avg_ms:.4f} ms")

        del tensor
        torch.cuda.empty_cache()

    results["all_reduce"] = allreduce_results

    # ── Test 3: Pipeline overhead simulation ──
    if rank == 0:
        print(f"\nPipeline send/recv chain simulation...")

    # Simulate activation passing through pipeline: rank 0 → 1 → 2 → ... → N-1
    seq_len = (exp_cfg.phase1.height // 16) * (exp_cfg.phase1.width // 16)
    dim = exp_cfg.model.dim
    activation = torch.randn(1, seq_len, dim, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(5):
        if rank < world_size - 1:
            dist.send(activation, dst=rank + 1)
        if rank > 0:
            dist.recv(activation, src=rank - 1)
        dist.barrier()

    # Measure end-to-end pipeline pass
    pipeline_times = []
    for _ in range(30):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()

        if rank < world_size - 1:
            dist.send(activation, dst=rank + 1)
        if rank > 0:
            dist.recv(activation, src=rank - 1)

        torch.cuda.synchronize()
        dist.barrier()
        pipeline_times.append((time.perf_counter() - start) * 1000)

    if rank == 0:
        avg_pipeline_ms = sum(pipeline_times) / len(pipeline_times)
        results["pipeline_pass"] = {
            "num_hops": world_size - 1,
            "avg_total_ms": round(avg_pipeline_ms, 4),
            "avg_per_hop_ms": round(avg_pipeline_ms / max(world_size - 1, 1), 4),
            "activation_size_mb": round(seq_len * dim * 2 / (1024 * 1024), 1),
        }
        print(f"  Pipeline pass ({world_size-1} hops): {avg_pipeline_ms:.4f} ms total, "
              f"{avg_pipeline_ms / max(world_size-1, 1):.4f} ms/hop")

    # ── Save results ──
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"COMMUNICATION PROFILING SUMMARY ({world_size}xH100 NVLink)")
        print(f"{'='*60}")

        out_path = os.path.join(paths.output_root, "phase4", "comm_profile",
                               f"comm_profile_{world_size}gpu.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {out_path}")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_configs.yaml")
    args = parser.parse_args()
    exp_cfg = OmegaConf.load(args.config)
    run_comm_profile(exp_cfg)


if __name__ == "__main__":
    main()
