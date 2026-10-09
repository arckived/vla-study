"Simulated ("fake") weight-only quantization for any PyTorch model."
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class QuantReport:
    bits: int
    group_size: int | None
    n_layers: int = 0
    n_params_quantized: int = 0
    n_params_total: int = 0
    est_bytes_before: int = 0
    est_bytes_after: int = 0
    mean_rel_error: float = 0.0  # mean ||W - Wq|| / ||W|| over quantized layers
    layers: list[str] = field(default_factory=list)

    def summary(self) -> str:
        frac = self.n_params_quantized / max(self.n_params_total, 1)
        saved = 1 - self.est_bytes_after / max(self.est_bytes_before, 1)
        return (
            f"[quant] {self.bits}-bit, group_size={self.group_size}: "
            f"{self.n_layers} Linear layers, {self.n_params_quantized/1e6:.1f}M of "
            f"{self.n_params_total/1e6:.1f}M params ({frac:.0%}); "
            f"est. weight memory {self.est_bytes_before/2**20:.0f} MiB -> "
            f"{self.est_bytes_after/2**20:.0f} MiB ({saved:.0%} smaller); "
            f"mean relative weight error {self.mean_rel_error:.4f}"
        )

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("layers")
        return d


def quantize_dequantize(w: torch.Tensor, bits: int, group_size: int | None = None) -> torch.Tensor:
    """Round a 2-D weight matrix to `bits` bits and return the dequantized values.

    For each row (or each group within a row):
        scale = max(|w|) / qmax          qmax = 2^(bits-1) - 1   (127 for 8-bit, 7 for 4-bit)
        q     = clamp(round(w / scale), -qmax-1, qmax)   (an integer)
        w_hat = q * scale                (what the model actually computes with)
    """
    if w.dim() != 2:
        raise ValueError(f"expected 2-D weight, got shape {tuple(w.shape)}")
    if not 2 <= bits <= 16:
        raise ValueError("bits must be between 2 and 16")

    orig_dtype = w.dtype
    w = w.float()
    out_f, in_f = w.shape
    qmax = 2 ** (bits - 1) - 1

    use_groups = group_size is not None and group_size < in_f and in_f % group_size == 0
    wg = w.reshape(out_f, in_f // group_size, group_size) if use_groups else w

    scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(wg / scale), -qmax - 1, qmax)
    w_hat = (q * scale).reshape(out_f, in_f)
    return w_hat.to(orig_dtype)


def _matches(name: str, include: str | None, exclude: str | None) -> bool:
    if include and not re.search(include, name):
        return False
    if exclude and re.search(exclude, name):
        return False
    return True


@torch.no_grad()
def apply_fake_weight_quant(
    model: nn.Module,
    bits: int,
    include: str | None = None,
    exclude: str | None = None,
    group_size: int | None = None,
    verbose: bool = True,
) -> QuantReport:
    """Fake-quantize the weights of every nn.Linear whose qualified name matches
    `include` (regex, None = all) and does not match `exclude`. Modifies in place.
    Biases, norms, embeddings and convolutions are left untouched, which mirrors
    what most weight-only LLM quantization methods do."""
    rep = QuantReport(bits=bits, group_size=group_size)
    rel_errors = []

    for p in model.parameters():
        rep.n_params_total += p.numel()
        rep.est_bytes_before += p.numel() * p.element_size()

    rep.est_bytes_after = rep.est_bytes_before
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or not _matches(name, include, exclude):
            continue
        w = mod.weight
        w_hat = quantize_dequantize(w.data, bits, group_size)
        rel_errors.append(((w.data.float() - w_hat.float()).norm() / w.data.float().norm().clamp(min=1e-12)).item())
        w.data.copy_(w_hat)

        n = w.numel()
        out_f, in_f = w.shape
        groups_per_row = in_f // group_size if (group_size and group_size < in_f and in_f % group_size == 0) else 1
        n_scales = out_f * groups_per_row
        rep.est_bytes_after -= n * w.element_size()          # remove full-precision weights
        rep.est_bytes_after += (n * bits + 7) // 8 + n_scales * 2  # N-bit weights + fp16 scales
        rep.n_layers += 1
        rep.n_params_quantized += n
        rep.layers.append(name)

    rep.mean_rel_error = sum(rel_errors) / len(rel_errors) if rel_errors else 0.0
    if rep.n_layers == 0:
        raise RuntimeError(
            f"No nn.Linear matched include={include!r} exclude={exclude!r}. "
            "Run `python -m vla_study.quantize --list <policy_path>` to see module names."
        )
    if verbose:
        print(rep.summary(), flush=True)
    return rep


def summarize_linear_modules(model: nn.Module, depth: int = 3) -> Counter:
    """Count Linear params grouped by the first `depth` components of their name.
    Use this to pick include/exclude regexes (e.g. VLM backbone vs. action expert)."""
    counts: Counter = Counter()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            prefix = ".".join(name.split(".")[:depth])
            counts[prefix] += mod.weight.numel()
    return counts


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect Linear layers of a LeRobot policy.")
    ap.add_argument("--list", metavar="POLICY_PATH", required=True,
                    help="Hub id or local path of a pretrained policy (e.g. outputs/train/.../pretrained_model)")
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    from vla_study.lerobot_compat import load_policy

    policy = load_policy(args.list, device="cpu")
    counts = summarize_linear_modules(policy, args.depth)
    total = sum(counts.values())
    print(f"{'module prefix':<70} {'Linear params':>14}  share")
    for prefix, n in counts.most_common():
        print(f"{prefix:<70} {n/1e6:>12.2f}M  {n/total:5.1%}")


if __name__ == "__main__":
    _cli()
