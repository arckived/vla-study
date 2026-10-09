# Run log

A record of every run behind the results in the README: where it ran, with what settings, and what it produced. Full outputs (checkpoints, evaluation files, rollout videos, logs) are archived in a private Hugging Face repository; the notebooks in this repo are kept output-free so they stay readable.

LeRobot commit used for all runs: `e0d50211ef236143ae867228662b7dfaba554f02`

| # | Date (2026) | Stage | Platform | Settings | Outcome |
|---|---|---|---|---|---|
| 1 | Oct 1 | Pipeline check (`baseline_smoke`) | Colab, NVIDIA L4 | Public pretrained SmolVLA LIBERO checkpoint, 1 episode per task | 9/10 tasks succeeded: rendering (EGL), cameras and action space wired correctly |
| 2 | Oct 1 | Training speed test (`speedtest`) | Colab, L4 | 200 steps, batch 16 | 2.38 steps/s using 4 GB of 23 GB GPU memory, so batch 32 chosen for the real runs |
| 3 | Oct 2 | Train model A (`train_A`) | Colab, L4 | Action expert only, 20k steps, batch 32, seed 1000 | ~5 h, about 2.3 epochs; final training loss 0.351 |
| 4 | Oct 4 | Evaluate A (`eval_trained`) | Colab, L4 | 10 tasks × 10 episodes | **65.0%** success, 95% CI [55.3, 73.6]; ~39 s per episode |
| 5 | Oct 5 | Train model B (`train_B`) | Colab, L4 | Action expert + language layers, otherwise identical to A | ~5.3 h; final training loss 0.322 |
| 6 | Oct 7 | Evaluate B (`eval_trained`) | Colab, L4 | 10 tasks × 10 episodes (first attempt interrupted by a disconnect, rerun from scratch) | **58.0%** success, 95% CI [48.2, 67.2] |
| 7 | Oct 7 | Inspect model / build paraphrases (`quant_list`, `paraphrase_generate`) | Kaggle, T4 | Module listing; 4 paraphrase levels + swapped control, reviewed by hand | VLM backbone ~301M params (75%), action expert ~98M (25%); all paraphrases kept the meaning |
| 8 | Oct 8 | Instruction tests (`paraphrase_eval`) | Kaggle, T4 (background run) | Model A, 10 tasks × 5 episodes per condition; every rewrite verified (0 unmatched) | original **66%**, synonym **26%**, restructured **0%**, verbose **0%**, swapped **6%** |

## Problems hit and how they were fixed

- **Plotting backend leaked into the evaluation environment.** Colab's notebook setting `MPLBACKEND=module://matplotlib_inline...` was inherited by the separate Python 3.12 environment, which doesn't have that module, so LIBERO crashed on import. Fixed by setting `MPLBACKEND=Agg` for all subprocesses.
- **LeRobot renamed a training option.** `--eval_freq` became `--env_eval_freq`. Diagnosed from the argument-parser error, which listed every unrecognized option at once, confirming this was the only one.
- **Python version.** Current LeRobot requires Python ≥ 3.12, so the notebooks build their own 3.12 environment with `uv` instead of using the platform's default Python.
- **Cloud sessions ending mid-run.** Checkpoints every 1,000 steps plus automatic Hugging Face backups every 15 minutes; a new session restores and resumes. Google Drive was avoided for checkpoints because it doesn't support the symbolic link LeRobot uses to mark the latest checkpoint.
- **Avoiding wrong evaluations.** The evaluation script only evaluates runs trained to the full step count and skips runs that already have results, so a half-trained model or an accidental re-run can't overwrite a result.

## Not run

- Weight quantization (`quant` stage). The tooling is implemented and the correct module filters were identified (run 7), but the evaluations were not run. Listed as future work.
