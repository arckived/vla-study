"""Back up experiment outputs to a private Hugging Face Hub repo, restore them in a new
session, and prune old checkpoints so the disk doesn't fill up.

"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
import traceback
from pathlib import Path

STEP_RE = re.compile(r"^\d+$")
MISC_PATTERNS = ["eval/*", "data/*", "results/*", "*.txt", "*.log", "*.json"]


# ---------- local checkpoint helpers (no network, unit-testable) ----------

def step_dirs(ckpt_root: Path) -> list[Path]:
    if not ckpt_root.is_dir():
        return []
    dirs = [p for p in ckpt_root.iterdir() if p.is_dir() and not p.is_symlink() and STEP_RE.match(p.name)]
    return sorted(dirs, key=lambda p: int(p.name))


def is_complete(step_dir: Path) -> bool:
    return (step_dir / "pretrained_model" / "train_config.json").exists() and (step_dir / "training_state").is_dir()


def latest_complete(ckpt_root: Path) -> Path | None:
    """The checkpoint `last` points to (LeRobot updates it after a save finishes),
    else the newest complete numbered folder."""
    last = ckpt_root / "last"
    if last.exists():
        target = last.resolve()
        if is_complete(target):
            return target
    for d in reversed(step_dirs(ckpt_root)):
        if is_complete(d):
            return d
    return None


def prune(outputs: Path, keep: int = 2) -> None:
    """Delete all but the newest `keep` checkpoints of each experiment (never the latest complete one)."""
    for ck in outputs.glob("train/*/checkpoints"):
        dirs = step_dirs(ck)
        protect = set(dirs[-keep:])
        lc = latest_complete(ck)
        if lc:
            protect.add(lc)
        for d in dirs:
            if d not in protect:
                shutil.rmtree(d, ignore_errors=True)
                print(f"[sync] pruned {d}", flush=True)


# ---------- Hub operations ----------

def _api():
    from huggingface_hub import HfApi
    return HfApi()


def push(repo: str, outputs: Path) -> None:
    api = _api()
    api.create_repo(repo, private=True, exist_ok=True)
    state_file = outputs / ".sync_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}

    for ck in sorted(outputs.glob("train/*/checkpoints")):
        exp = ck.parent.name
        lc = latest_complete(ck)
        if lc is None or state.get(exp) == lc.name:
            continue
        print(f"[sync] uploading {exp} checkpoint {lc.name} ...", flush=True)
        api.upload_folder(repo_id=repo, folder_path=str(lc), path_in_repo=f"train/{exp}/checkpoint_last",
                          delete_patterns="*", commit_message=f"{exp} checkpoint {lc.name}")
        api.upload_file(repo_id=repo, path_or_fileobj=lc.name.encode(), path_in_repo=f"train/{exp}/LAST_STEP.txt")
        state[exp] = lc.name
        state_file.write_text(json.dumps(state))

    api.upload_folder(repo_id=repo, folder_path=str(outputs), path_in_repo=".",
                      allow_patterns=MISC_PATTERNS, ignore_patterns=["train/*", ".sync_state.json"],
                      commit_message="sync eval/data/results")
    print(f"[sync] push done {time.strftime('%H:%M:%S')}", flush=True)


def pull(repo: str, outputs: Path) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    outputs.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download(repo_id=repo, local_dir=tmp)
            tmp = Path(tmp)
            # small files: copy anything we don't already have
            for src in tmp.rglob("*"):
                rel = src.relative_to(tmp)
                if src.is_dir() or rel.parts[0] in ("train", ".cache") or rel.name == ".gitattributes":
                    continue
                dst = outputs / rel
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            # checkpoints: rebuild outputs/train/<exp>/checkpoints/<step> and the `last` link
            state = {}
            for step_file in tmp.glob("train/*/LAST_STEP.txt"):
                exp = step_file.parent.name
                step = step_file.read_text().strip()
                ck = outputs / "train" / exp / "checkpoints"
                dst = ck / step
                if not dst.exists():
                    shutil.copytree(step_file.parent / "checkpoint_last", dst)
                last = ck / "last"
                if last.is_symlink() or last.exists():
                    last.unlink()
                last.symlink_to(step)  # relative link, like LeRobot's
                state[exp] = step
                print(f"[sync] restored {exp} at step {step}", flush=True)
            (outputs / ".sync_state.json").write_text(json.dumps(state))
    except RepositoryNotFoundError:
        print(f"[sync] {repo} does not exist yet - nothing to restore (first session).", flush=True)


def loop(repo: str, outputs: Path, every: int, keep: int) -> None:
    while True:
        try:
            prune(outputs, keep)
            push(repo, outputs)
        except Exception:  # never let a network hiccup kill the backup loop
            traceback.print_exc()
        time.sleep(every)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "pull", "loop", "prune"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--outputs", required=True, type=Path)
    ap.add_argument("--every", type=int, default=900, help="seconds between backups (loop)")
    ap.add_argument("--keep", type=int, default=2, help="checkpoints to keep locally per experiment")
    a = ap.parse_args()
    if a.cmd != "prune" and not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set")
    {"push": lambda: push(a.repo, a.outputs), "pull": lambda: pull(a.repo, a.outputs),
     "prune": lambda: prune(a.outputs, a.keep), "loop": lambda: loop(a.repo, a.outputs, a.every, a.keep)}[a.cmd]()


if __name__ == "__main__":
    main()
