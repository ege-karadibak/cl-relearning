"""Analysis: turn the logs into the experiment's metrics. Reads logs only, never checkpoints.

    from metrics import load_stage3, curve_metrics, summary, memory_effect
    load_stage3()                   ResNet-18 runs (logs/)
    load_stage3(LOGS / "vit")       ViT runs (logs/vit/); every loader takes the folder the same way

Stage-3 curves are noisy (constant learning rate on 5k images: test accuracy moves several
points between measurements), so every metric below uses a SMOOTHED curve: each point is the
mean of itself and the previous `window - 1` measurements (default 3). Raw curves are kept for plots.

Metrics per stage-3 run:
    steps_to_50 / 70 / 80   first step where the smoothed accuracy reaches 50 / 70 / 80 %
                            (NaN = not reached within the budget: slower than every run that did)
    auc                     area under the raw accuracy curve, divided by the number of steps
                            (= average accuracy over the whole relearning)
    final_acc               mean raw accuracy over the last 500 steps
    gain_after_50           accuracy gained in the 500 steps after first reaching 50 % (aligned curves:
                            "does it learn faster, or did it only start ahead?")
"""

from pathlib import Path                     # file paths
import json                                  # log lines are JSON
import numpy as np                           # trapezoid area, NaN
import pandas as pd                          # tables

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
ARMS = ["A", "B", "C"]
SUBSETS = ["a_seen", "never_seen"]
THRESHOLDS = (0.5, 0.7, 0.8)


# ---------------------------------------------------------------- loading
def read_log(path):
    """One .jsonl log → DataFrame (one row per measurement)."""
    return pd.DataFrame([json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()])


def load_stage(stage, folder=LOGS):
    """All runs of one stage (1, 2 or 3) as one long DataFrame with arm / seed (/ subset) columns."""
    frames = []
    for path in sorted(Path(folder).glob(f"seed*/*_stage{stage}*.jsonl")):
        name = path.stem                                   # e.g. "A_stage3_never_seen", "A_stage1_probe"
        if name.endswith("_probe"):
            continue
        df = read_log(path)
        df["seed"] = int(path.parent.name.removeprefix("seed"))
        df["arm"] = name[0]
        if stage == 3:
            df["subset"] = name.split(f"_stage3_")[1]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no stage-{stage} logs under {folder}; copy them from the server first")
    return pd.concat(frames, ignore_index=True)


def load_stage3(folder=LOGS):
    return load_stage(3, folder)


def load_probes(folder=LOGS):
    """Probe results after stage 1 (and B at stage 0) from the *_probe.jsonl files."""
    rows = [json.loads(p.read_text().splitlines()[-1]) for p in sorted(Path(folder).glob("seed*/*_probe.jsonl"))]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- one curve
def smooth(acc, window=3):
    """Mean of each point and the previous window-1 points (fewer at the very start)."""
    return pd.Series(acc).rolling(window, min_periods=1).mean().to_numpy()


def first_step_at(steps, acc_smooth, threshold):
    """First step where the smoothed accuracy reaches the threshold; NaN if never."""
    hit = np.flatnonzero(acc_smooth >= threshold)
    return float(steps[hit[0]]) if len(hit) else np.nan


def curve_metrics(run, acc_col="test_acc", window=3, final_span=500, aligned_span=500):
    """All stage-3 metrics for one run (DataFrame sorted by step)."""
    run = run.sort_values("step")
    steps, acc = run["step"].to_numpy(dtype=float), run[acc_col].to_numpy(dtype=float)
    acc_s = smooth(acc, window)
    out = {f"steps_to_{int(t * 100)}": first_step_at(steps, acc_s, t) for t in THRESHOLDS}
    out["auc"] = float(np.trapezoid(acc, steps) / (steps[-1] - steps[0]))
    out["final_acc"] = float(acc[steps >= steps[-1] - final_span].mean())
    start = out["steps_to_50"]
    if np.isnan(start):
        out["gain_after_50"] = np.nan
    else:
        later = acc_s[steps >= start + aligned_span]
        out["gain_after_50"] = float(later[0] - acc_s[steps == start][0]) if len(later) else np.nan
    return out


# ---------------------------------------------------------------- tables
def per_run(stage3):
    """One row per (arm, seed, subset) with every metric."""
    rows = []
    for (arm, seed, subset), run in stage3.groupby(["arm", "seed", "subset"]):
        rows.append({"arm": arm, "seed": seed, "subset": subset, **curve_metrics(run)})
    return pd.DataFrame(rows)


def summary(runs):
    """Mean and standard deviation across seeds, per arm and subset.

    'not reached' (NaN) steps-to-threshold are counted in n_reached instead of being averaged.
    """
    metrics = [c for c in runs.columns if c not in ("arm", "seed", "subset")]
    g = runs.groupby(["subset", "arm"])
    table = g[metrics].agg(["mean", "std"])
    for t in THRESHOLDS:
        col = f"steps_to_{int(t * 100)}"
        table[(col, "n_reached")] = g[col].apply(lambda s: int(s.notna().sum()))
    return table.sort_index(axis=1)


def differences(runs, metric, a="A", b="C", subset="never_seen"):
    """Per-seed difference a − b in one metric, plus its mean and std (the noise across seeds)."""
    r = runs[runs["subset"] == subset].pivot(index="seed", columns="arm", values=metric)
    d = r[a] - r[b]
    return d, float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else np.nan


def memory_effect(runs, metric="auc"):
    """gap(arm) = metric on A-seen − metric on never-seen, per seed.

    memory effect = gap(A) − gap(B), and gap(A) − gap(C). Positive = A remembers the specific
    images it trained on, beyond the A-seen subset simply being easier (which B and C measure).
    For steps-to-threshold metrics, lower is better, so the sign is flipped.
    """
    sign = -1.0 if metric.startswith("steps_to") else 1.0
    r = runs.pivot_table(index=["seed", "arm"], columns="subset", values=metric)
    gap = sign * (r["a_seen"] - r["never_seen"])
    gap = gap.unstack("arm")
    out = pd.DataFrame({"gap_A": gap["A"], "gap_B": gap["B"], "gap_C": gap["C"]})
    out["effect_vs_B"] = out["gap_A"] - out["gap_B"]
    out["effect_vs_C"] = out["gap_A"] - out["gap_C"]
    return out


def mean_curves(stage3, acc_col="test_acc", window=3):
    """Mean ± std across seeds of the smoothed curve, per arm and subset (for plots)."""
    s = stage3.sort_values("step").copy()
    s["acc_s"] = s.groupby(["arm", "seed", "subset"])[acc_col].transform(lambda x: smooth(x.to_numpy(), window))
    return s.groupby(["subset", "arm", "step"])["acc_s"].agg(["mean", "std"]).reset_index()


def aligned_curves(stage3, start_at=0.5, acc_col="test_acc", window=3, span=1500, grid_every=25):
    """Shift every run so step 0 is where its smoothed accuracy first reaches `start_at`.

    Runs reach `start_at` at different steps, so each shifted curve is interpolated onto one
    common grid (0, 25, 50, ... span) before averaging. Runs that never reach it are left out.
    """
    grid = np.arange(0, span + 1, grid_every, dtype=float)
    frames = []
    for (arm, seed, subset), run in stage3.groupby(["arm", "seed", "subset"]):
        run = run.sort_values("step")
        steps = run["step"].to_numpy(dtype=float)
        acc_s = smooth(run[acc_col].to_numpy(dtype=float), window)
        t0 = first_step_at(steps, acc_s, start_at)
        if np.isnan(t0):
            continue
        g = grid[grid <= steps[-1] - t0]                       # only where the run has data
        frames.append(pd.DataFrame({"arm": arm, "seed": seed, "subset": subset,
                                    "step_from_start": g, "acc_s": np.interp(g + t0, steps, acc_s)}))
    a = pd.concat(frames, ignore_index=True)
    return a.groupby(["subset", "arm", "step_from_start"])["acc_s"].agg(["mean", "std"]).reset_index()
