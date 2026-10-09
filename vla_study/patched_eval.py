"""Run LeRobot's normal closed-loop evaluation, with two optional interventions:

  1. -quant-bits N        fake-quantize the policy's Linear weights right after loading
  2. -paraphrase-file F   replace each task instruction with a paraphrase before the
                           policy's preprocessor tokenizes it

Everything is passed unchanged to lerobot-eval, e.g.

  python -m vla_study.patched_eval -quant-bits 4 -quant-include "vlm"  \
      -policy.path=outputs/train/expert_only/checkpoints/last/pretrained_model \
      -env.type=libero -env.task=libero_object -eval.n_episodes=10 -eval.batch_size=1 \
      -output_dir=outputs/eval/expert_only_q4_vlm

How it works (monkey-patching):
    lerobot_eval does `from lerobot.policies.factory import make_policy, ...` at import
    time, so those names live inside the lerobot_eval module. We swap them for wrapped
    versions *in that module*, then call its main(). LeRobot's own rollout, success
    counting and video saving are untouched, so results stay comparable to plain
    lerobot-eval runs.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import sys

from vla_study.lerobot_compat import import_eval_module


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


class TaskRewritingPreprocessor:
    """Wraps a policy preprocessor; swaps data['task'] using a {original: paraphrase} map."""

    def __init__(self, inner, mapping: dict[str, str]):
        self.inner = inner
        self.mapping = {normalize(k): v for k, v in mapping.items()}
        self.hits = 0
        self.misses: set[str] = set()
        self.shown = 0
        atexit.register(self.report)

    def _rewrite_one(self, t: str) -> str:
        new = self.mapping.get(normalize(t))
        if new is None:
            self.misses.add(t)
            return t
        self.hits += 1
        if self.shown < 3:
            print(f"[paraphrase] '{t}'  ->  '{new}'", flush=True)
            self.shown += 1
        return new

    def __call__(self, data, *args, **kwargs):
        if isinstance(data, dict) and "task" in data:
            data = dict(data)
            t = data["task"]
            data["task"] = self._rewrite_one(t) if isinstance(t, str) else [self._rewrite_one(x) for x in t]
        return self.inner(data, *args, **kwargs)

    def __getattr__(self, name):  # behave like the wrapped processor for everything else
        return getattr(self.inner, name)

    def report(self):
        print(f"[paraphrase] rewritten instructions: {self.hits}; unmatched: {len(self.misses)}", flush=True)
        if self.hits == 0:
            print("[paraphrase] WARNING: nothing was rewritten - this run is NOT a paraphrase test. "
                  "Check that the keys in your paraphrase file match the env's task strings.", flush=True)
        for m in sorted(self.misses)[:10]:
            print(f"[paraphrase]   unmatched: {m!r}", flush=True)


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: python -m vla_study.patched_eval [options] -- <lerobot-eval args>")
    split = argv.index("--")
    ours, theirs = argv[:split], argv[split + 1:]

    ap = argparse.ArgumentParser()
    ap.add_argument("--quant-bits", type=int, default=None)
    ap.add_argument("--quant-include", default=None, help="regex on module names (default: all Linear)")
    ap.add_argument("--quant-exclude", default=None)
    ap.add_argument("--quant-group-size", type=int, default=None, help="e.g. 128 for 4-bit")
    ap.add_argument("--paraphrase-file", default=None)
    ap.add_argument("--paraphrase-level", default=None, help="key inside the paraphrase file")
    args = ap.parse_args(ours)

    ev = import_eval_module()
    meta = {"quant": None, "paraphrase_level": args.paraphrase_level}

    if args.quant_bits:
        from vla_study.quantize import apply_fake_weight_quant

        if not hasattr(ev, "make_policy"):
            sys.exit("lerobot_eval has no `make_policy`; update vla_study/patched_eval.py for your LeRobot version")
        orig_make_policy = ev.make_policy

        def make_policy_quantized(*a, **k):
            policy = orig_make_policy(*a, **k)
            rep = apply_fake_weight_quant(policy, args.quant_bits, args.quant_include,
                                          args.quant_exclude, args.quant_group_size)
            meta["quant"] = rep.to_dict() | {"include": args.quant_include, "exclude": args.quant_exclude}
            return policy

        ev.make_policy = make_policy_quantized

    if args.paraphrase_file:
        if not args.paraphrase_level:
            sys.exit("--paraphrase-level is required with --paraphrase-file")
        with open(args.paraphrase_file) as f:
            mapping = json.load(f)[args.paraphrase_level]
        if not hasattr(ev, "make_pre_post_processors"):
            sys.exit("lerobot_eval has no `make_pre_post_processors`; update patched_eval.py")
        orig_make_proc = ev.make_pre_post_processors

        def make_proc_rewriting(*a, **k):
            pre, post = orig_make_proc(*a, **k)
            return TaskRewritingPreprocessor(pre, mapping), post

        ev.make_pre_post_processors = make_proc_rewriting

    # Save what we changed next to LeRobot's own eval output, for results.py
    out_dir = next((x.split("=", 1)[1] for x in theirs if x.startswith("--output_dir=")), None)

    def save_meta():
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "study_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
    atexit.register(save_meta)

    sys.argv = [sys.argv[0]] + theirs
    entry = getattr(ev, "main", None) or getattr(ev, "eval_main")
    entry()


if __name__ == "__main__":
    main()
