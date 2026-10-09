"""Collect every lerobot-eval run under outputs/eval/ into one results table + chart.

    python -m vla_study.results -eval-root outputs/eval -out results

Writes results/results.md (paste into your README) and results/success_rates.png.

Each success rate gets a 95% Wilson confidence interval. With 100 episodes, a
measured 80% really means "somewhere between ~71% and ~87%", so two configs whose
intervals overlap heavily are NOT meaningfully different. Saying this in an
interview signals research maturity.
"""
from __future__ import annotations

import argparse
import json
import math
import os


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def find_overall(info: dict) -> dict | None:
    """LeRobot's eval_info.json layout has changed over versions; search for the
    aggregate block that contains 'pc_success'."""
    for key in ("overall", "aggregated"):
        if isinstance(info.get(key), dict) and "pc_success" in info[key]:
            return info[key]
    if "pc_success" in info:
        return info
    for v in info.values():
        if isinstance(v, dict):
            found = find_overall(v)
            if found:
                return found
    return None


def count_episodes(info: dict, overall: dict) -> int | None:
    for k in ("n_episodes", "num_episodes"):
        if isinstance(overall.get(k), int):
            return overall[k]
    eps = info.get("per_episode")
    return len(eps) if isinstance(eps, list) else None


def collect(eval_root: str) -> list[dict]:
    rows = []
    for dirpath, _dirs, files in os.walk(eval_root, followlinks=True):
        # LeRobot normally writes eval_info.json; fall back to any other JSON in the folder
        candidates = ["eval_info.json"] if "eval_info.json" in files else \
            [f for f in files if f.endswith(".json") and f != "study_meta.json"]
        info = None
        for name in candidates:
            try:
                with open(os.path.join(dirpath, name)) as f:
                    loaded = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(loaded, dict) and find_overall(loaded):
                info = loaded
                break
        if info is None:
            continue
        overall = find_overall(info)
        if overall is None:
            print(f"skip {dirpath}: no pc_success found")
            continue
        n = count_episodes(info, overall)
        pc = float(overall["pc_success"])
        lo, hi = wilson_interval(pc / 100 * n, n) if n else (float("nan"), float("nan"))
        meta = {}
        if os.path.exists(os.path.join(dirpath, "study_meta.json")):
            with open(os.path.join(dirpath, "study_meta.json")) as f:
                meta = json.load(f)
        rows.append({"run": os.path.relpath(dirpath, eval_root), "success": pc, "n": n,
                     "ci_lo": lo * 100, "ci_hi": hi * 100,
                     "est_mem_mib": (meta.get("quant") or {}).get("est_bytes_after", 0) / 2**20 or None,
                     "paraphrase": meta.get("paraphrase_level")})
    return sorted(rows, key=lambda r: r["run"])


def write_markdown(rows: list[dict], path: str) -> None:
    lines = ["| Run | Success % | 95% CI | Episodes | Est. weight MiB | Paraphrase |",
             "|---|---:|---|---:|---:|---|"]
    for r in rows:
        mem = f"{r['est_mem_mib']:.0f}" if r["est_mem_mib"] else "-"
        lines.append(f"| {r['run']} | {r['success']:.1f} | [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}] | "
                     f"{r['n'] or '?'} | {mem} | {r['paraphrase'] or '-'} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot(rows: list[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["run"] for r in rows]
    vals = [r["success"] for r in rows]
    err = [[v - r["ci_lo"] for v, r in zip(vals, rows)], [r["ci_hi"] - v for v, r in zip(vals, rows)]]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(rows)), 4))
    ax.bar(range(len(rows)), vals, yerr=err, capsize=4, color="#4C72B0")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Task success (%)")
    ax.set_ylim(0, 100)
    ax.set_title("LIBERO closed-loop success (95% Wilson CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="outputs/eval")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rows = collect(a.eval_root)
    if not rows:
        print(f"No finished evaluations found under {a.eval_root!r}.")
        if os.path.isdir(a.eval_root):
            print("Folders present:", sorted(os.listdir(a.eval_root)) or "(none)")
            print("A folder without eval_info.json usually means that eval crashed or was interrupted.")
        else:
            print("That folder does not exist yet: run the evaluation stages (01, 04, 05, 06) first.")
        raise SystemExit(1)
    os.makedirs(a.out, exist_ok=True)
    write_markdown(rows, os.path.join(a.out, "results.md"))
    plot(rows, os.path.join(a.out, "success_rates.png"))
    print(open(os.path.join(a.out, "results.md")).read())


if __name__ == "__main__":
    main()
