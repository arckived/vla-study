"""Fast, simulator-free quantization study.

Closed-loop success rate (patched_eval.py) is the metric that matters, but each run
takes a long time. This script answers a cheaper question in minutes:
    "On real dataset frames, how far do the quantized policy's predicted action
     chunks drift from the full-precision policy's?"

IMPORTANT DETAIL - SmolVLA is stochastic. It generates actions with flow matching,
starting from random noise. Two identical models give different outputs unless the
noise is identical, so we reseed the RNG with the same value before every call.
Without this, you would be measuring noise, not quantization error.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

from vla_study.lerobot_compat import load_policy, load_processors
from vla_study.quantize import apply_fake_weight_quant


def to_batch(item: dict) -> dict:
    """Turn one dataset sample into a batch of size 1."""
    batch = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0)
        elif isinstance(v, str):
            batch[k] = [v]
        else:
            batch[k] = v
    return batch


@torch.no_grad()
def predict_chunk(policy, batch: dict, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(policy, "predict_action_chunk"):
        out = policy.predict_action_chunk(batch)
    else:  # older versions: select_action returns one step
        policy.reset()
        out = policy.select_action(batch)
    return out.float().cpu()


def parse_config(s: str):
    name, bits, *rest = s.split(":")
    include = rest[0] if rest and rest[0] else None
    return name, (None if bits == "none" else int(bits)), include


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--dataset", default="lerobot/libero")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--group-size", type=int, default=128, help="used for bits <= 4")
    ap.add_argument("--latency-iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/offline_bench.json")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.dataset)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), k=min(args.n_samples, len(dataset)))
    print(f"dataset {args.dataset}: {len(dataset)} frames, sampling {len(indices)}")

    reference: list[torch.Tensor] | None = None
    results = []
    for cfg in args.configs:
        name, bits, include = parse_config(cfg)
        print(f"\n=== {name} (bits={bits}, include={include!r}) ===")
        policy = load_policy(args.policy_path, device=args.device)
        pre, _post = load_processors(policy, args.policy_path)

        qrep = None
        if bits is not None:
            gs = args.group_size if bits <= 4 else None
            qrep = apply_fake_weight_quant(policy, bits, include=include, group_size=gs).to_dict()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        chunks = []
        for j, idx in enumerate(indices):
            batch = pre(to_batch(dataset[idx]))
            chunks.append(predict_chunk(policy, batch, seed=args.seed + j))

        # latency on the first sample, after warm-up
        batch0 = pre(to_batch(dataset[indices[0]]))
        for _ in range(3):
            predict_chunk(policy, batch0, seed=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.latency_iters):
            predict_chunk(policy, batch0, seed=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / args.latency_iters * 1000

        row = {"name": name, "bits": bits, "include": include, "latency_ms_per_chunk": latency_ms,
               "peak_cuda_mem_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
               "quant": qrep}
        if reference is None:
            if bits is not None:
                print("WARNING: first config should be full precision ('fp:none') - it is the reference.")
            reference = chunks
            row["mse_vs_reference"] = 0.0
        else:
            mse = torch.stack([((c - r) ** 2).mean() for c, r in zip(chunks, reference)])
            row["mse_vs_reference"] = mse.mean().item()
            row["mse_vs_reference_p95"] = mse.quantile(0.95).item()
        print({k: v for k, v in row.items() if k != "quant"})
        results.append(row)
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    print("Note: latency is identical across fake-quant configs by design (weights are stored at "
          "full precision). Report estimated memory from the 'quant' field, not measured speed-ups.")


if __name__ == "__main__":
    main()
