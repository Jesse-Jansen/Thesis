"""Shared helpers that do not need a GPU.

Contains the experiment configuration (paths, conditions), the split checks,
the two reweighting schemes, the metrics and the aggregation over seeds
including the bootstrap. It is imported by training.py, ddi.py and the
notebooks. Nothing in this file needs PyTorch, so the result tables can be
rebuilt on a machine without a GPU.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve


# Paths

# Expected layout (the defaults assume this repository sits inside the project folder):
#
#   <project>/data/finalfitz17k/           Fitzpatrick17k images, named <md5hash>.jpg
#   <project>/binary_experiment_splits/    train.csv, val.csv, test.csv (see make_splits.py)
#   <project>/<this repository>/           code, summary/ and figures/; by default also the run folders
#
# If your folders are somewhere else, set the environment variables used below, or
# put them in a file local_paths.json next to this file (see the README).
REPO_DIR = Path(__file__).resolve().parent


def path_setting(name, default):
    """Folder for a setting: environment variable first, then local_paths.json, then the default."""
    if name in os.environ:
        return Path(os.environ[name])
    local_file = REPO_DIR / "local_paths.json"
    if local_file.is_file():
        value = json.loads(local_file.read_text(encoding="utf-8")).get(name)
        if value:
            return Path(value)
    return Path(default)


# Where the run folders  are read from and written to.
EXPERIMENTS_DIR = path_setting("THESIS_EXPERIMENTS_DIR", REPO_DIR)
PROJECT_DIR = path_setting("FITZPATRICK17K_DIR", EXPERIMENTS_DIR.parent)
IMAGE_DIR = PROJECT_DIR / "data" / "finalfitz17k"
SPLIT_DIR = PROJECT_DIR / "binary_experiment_splits"
# Result tables and figures always go into the repository.
SUMMARY_DIR = REPO_DIR / "summary"
FIGURES_DIR = REPO_DIR / "figures"

# SHA-256 of the three split files. load_splits() checks them, so every condition
# and every seed is trained and tested on exactly the same images.
EXPECTED_SPLIT_SHA256 = {
    "train": "064fdbe82d9f917817dcf5017d5c307f20f32861142620375b1277a44e490939",
    "val": "ef892fa647000e63fd2c2f212505b06ad6bf3b70be864d6d84ee483c2216e0ab",
    "test": "1f27b814e5aab671c9e414318d5c00e80f1820ac654979f0eb46d03c9c71f8a5",
}

FST_GROUPS = ["FST 1-2", "FST 3-4", "FST 5-6"]
TARGET_FST_TYPES = (5, 6)  # skin types that get the targeted colour augmentation
DEFAULT_SEEDS = [42, 43, 44]
FIXED_THRESHOLD = 0.5

# The six conditions of the thesis. A condition only sets the colour mode and the
# loss weighting, everything else is shared (see training.HP).
CONDITIONS = {
    "baseline": {
        "folder": "01_baseline",
        "colour_mode": "none",
        "weighting": "none",
        "description": "Spatial augmentation only (resize, random resized crop, horizontal flip).",
    },
    "universal_colour": {
        "folder": "02_universal_colour",
        "colour_mode": "universal",
        "weighting": "none",
        "description": "Mild colour jitter (p=0.5) applied to all training images.",
    },
    "targeted_colour": {
        "folder": "03_targeted_colour",
        "colour_mode": "targeted",
        "weighting": "none",
        "description": "Mild colour jitter (p=0.5) applied only to FST V-VI training images.",
    },
    "reweighting_effective_number": {
        "folder": "04_reweighting_effective_number",
        "colour_mode": "none",
        "weighting": "effective_number",
        "description": "Class-balanced loss weights per FST group, effective number of samples (Cui et al., 2019), beta=0.999.",
    },
    "reweighting_kamiran": {
        "folder": "05_reweighting_kamiran",
        "colour_mode": "none",
        "weighting": "kamiran",
        "description": "Reweighing of Kamiran & Calders (2012): w = n_s*n_y / (N*n_sy) over FST types 1-6 x label.",
    },
    "combined": {
        "folder": "06_combined_targeted_colour_effective_number",
        "colour_mode": "targeted",
        "weighting": "effective_number",
        "description": "Targeted mild colour jitter (FST V-VI) plus effective-number weights per FST group.",
    },
}

# Statistics computed for every model in the bootstrap, in this order.
AUC_STATS = ["auc", "auc_fst_1_2", "auc_fst_3_4", "auc_fst_5_6", "auc_gap_max_min", "auc_gap_1_2_minus_5_6"]
# The subset that is reported in the thesis tables.
REPORTED_STATS = ["auc", "auc_fst_1_2", "auc_fst_3_4", "auc_fst_5_6", "auc_gap_1_2_minus_5_6"]


def condition_dir(condition):
    """Folder that holds the notebook and the seed_<n> run folders of a condition."""
    return EXPERIMENTS_DIR / CONDITIONS[condition]["folder"]


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def add_fst_group(frame):
    """Add the column fst_group (FST 1-2, 3-4 or 5-6) based on fitzpatrick_scale."""
    frame = frame.copy()
    frame["fst_group"] = pd.cut(
        frame["fitzpatrick_scale"], bins=[0, 2, 4, 6], labels=FST_GROUPS
    ).astype(str)
    return frame


def load_splits(split_dir=SPLIT_DIR, image_dir=IMAGE_DIR, check_images=True):
    """Load the train, validation and test split and check that they are usable.

    The checks cover the file hashes, the labels, the skin types, overlap between
    the splits and (optionally) whether every image file exists.

    Returns (train, val, test, hashes), where hashes maps the split name to the
    SHA-256 of its csv file.
    """
    split_dir, image_dir = Path(split_dir), Path(image_dir)
    frames, hashes = {}, {}
    for name in ["train", "val", "test"]:
        raw = (split_dir / f"{name}.csv").read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        assert hashes[name] == EXPECTED_SPLIT_SHA256[name], (
            f"{name}.csv is not the split file that was used for the thesis."
        )
        frame = add_fst_group(pd.read_csv(split_dir / f"{name}.csv"))
        assert not frame.empty
        assert frame["md5hash"].notna().all() and frame["md5hash"].is_unique
        assert frame["binary_label"].isin([0, 1]).all()
        assert frame["fitzpatrick_scale"].isin([1, 2, 3, 4, 5, 6]).all()
        assert frame["binary_label"].nunique() == 2
        frames[name] = frame

    ids = {name: set(frame.md5hash) for name, frame in frames.items()}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])

    if check_images:
        missing = [h for h in set.union(*ids.values()) if not (image_dir / f"{h}.jpg").is_file()]
        assert not missing, f"Missing images: {missing[:5]}"
    return frames["train"], frames["val"], frames["test"], hashes


# ---------------------------------------------------------------------------
# Sample weights (always computed from the training split only)
# ---------------------------------------------------------------------------
def effective_number_weights(train_df, beta=0.999, group_col="fst_group"):
    """Effective-number weights per FST group (Cui et al., 2019).

    The weight of group g is proportional to (1 - beta) / (1 - beta ** n_g), with
    n_g the number of training images in that group. The weights are rescaled so
    that the mean weight over all training images is 1. This keeps the size of the
    loss, and with it the effective learning rate, comparable to the baseline.

    Returns the weight per training image and a small table for the logs.
    """
    counts = train_df[group_col].value_counts().reindex(FST_GROUPS)
    assert counts.notna().all() and (counts > 0).all()
    raw = (1.0 - beta) / (1.0 - np.power(beta, counts.astype(float)))
    per_image = train_df[group_col].map(raw)
    scale = 1.0 / per_image.mean()
    weights = (raw * scale).to_dict()
    sample_weight = train_df[group_col].map(weights).to_numpy(dtype=float)
    table = pd.DataFrame({
        "fst_group": FST_GROUPS,
        "n_train": [int(counts[g]) for g in FST_GROUPS],
        "effective_number": [float((1 - beta ** counts[g]) / (1 - beta)) for g in FST_GROUPS],
        "weight": [float(weights[g]) for g in FST_GROUPS],
    })
    return sample_weight, table


def kamiran_weights(train_df):
    """Reweighing of Kamiran & Calders (2012) over skin type (1-6) and label.

    w(s, y) = n_s * n_y / (N * n_sy). With these weights skin type and label are
    statistically independent in the weighted training set. The mean weight is 1
    by construction.

    Returns the weight per training image and a table with the weight per cell.
    """
    counts = pd.crosstab(train_df.fitzpatrick_scale, train_df.binary_label).reindex(
        index=range(1, 7), columns=[0, 1], fill_value=0
    )
    assert (counts.to_numpy() > 0).all(), "Every FST type x label cell needs training images."
    n = len(train_df)
    ns, ny = counts.sum(axis=1), counts.sum(axis=0)
    weight_map = {
        (s, y): float(ns.loc[s] * ny.loc[y] / (n * counts.loc[s, y]))
        for s in counts.index for y in counts.columns
    }
    sample_weight = np.array([
        weight_map[(int(s), int(y))]
        for s, y in zip(train_df.fitzpatrick_scale, train_df.binary_label)
    ])
    table = pd.DataFrame([
        {"fst_type": s, "binary_label": y, "n_train": int(counts.loc[s, y]), "weight": weight_map[(s, y)]}
        for s in range(1, 7) for y in [0, 1]
    ])
    return sample_weight, table


def make_sample_weights(train_df, weighting):
    """Weight per training image for the given scheme ("none", "effective_number" or "kamiran")."""
    if weighting == "none":
        return np.ones(len(train_df)), None
    if weighting == "effective_number":
        weights, table = effective_number_weights(train_df)
    elif weighting == "kamiran":
        weights, table = kamiran_weights(train_df)
    else:
        raise ValueError(f"Unknown weighting: {weighting}")
    assert np.isclose(weights.mean(), 1.0), weights.mean()
    return weights, table


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def youden_threshold(y_true, y_prob):
    """Threshold that maximises Youden's index (sensitivity + specificity - 1).

    Always call this with validation data, never with test data.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best = np.argmax(tpr - fpr)
    return float(min(thresholds[best], 1.0))


def calculate_metrics(y_true, y_prob, threshold):
    """AUC plus the threshold-based metrics for one set of predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    both = np.unique(y_true).size == 2  # AUC is undefined when only one class is present
    return {
        "n": len(y_true),
        "n_benign": int((y_true == 0).sum()),
        "n_malignant": int((y_true == 1).sum()),
        "auc": roc_auc_score(y_true, y_prob) if both else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "malignant_recall": sens,
        "specificity": spec,
        "balanced_accuracy": (sens + spec) / 2 if both else np.nan,
        "malignant_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0) if (y_true == 1).any() else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


def evaluate_predictions(test_pred, thresholds):
    """Metrics for the whole test set, per FST group, and the gaps between groups.

    thresholds is a dict with one entry per threshold rule, for example
    {"fixed_0.5": 0.5, "val_youden": 0.43}. Returns three tables: overall,
    per FST group, and the differences between the groups.
    """
    overall, by_group, gaps = [], [], []
    for rule, thr in thresholds.items():
        overall.append({"threshold_rule": rule, **calculate_metrics(test_pred.binary_label, test_pred.predicted_prob, thr)})
        rows = []
        for group in FST_GROUPS:
            sub = test_pred[test_pred.fst_group == group]
            rows.append({"threshold_rule": rule, "fst_group": group,
                         **calculate_metrics(sub.binary_label, sub.predicted_prob, thr)})
        by_group += rows
        g = pd.DataFrame(rows).set_index("fst_group")
        for metric in ["auc", "malignant_recall", "specificity", "balanced_accuracy"]:
            v = g[metric]
            gaps.append({
                "threshold_rule": rule, "metric": metric,
                "fst_1_2": v["FST 1-2"], "fst_3_4": v["FST 3-4"], "fst_5_6": v["FST 5-6"],
                "max_min_gap": v.max() - v.min(),
                "gap_1_2_minus_5_6": v["FST 1-2"] - v["FST 5-6"],
            })
    return pd.DataFrame(overall), pd.DataFrame(by_group), pd.DataFrame(gaps)


# ---------------------------------------------------------------------------
# Collecting the finished runs
# ---------------------------------------------------------------------------
def completed_runs(experiments_dir=EXPERIMENTS_DIR):
    """Table with one row per finished run: condition, seed and run folder.

    A run counts as finished when run_complete.json exists in its folder.
    """
    runs = []
    for condition, spec in CONDITIONS.items():
        folder = Path(experiments_dir) / spec["folder"]
        for marker in sorted(folder.glob("seed_*/run_complete.json")):
            info = json.loads(marker.read_text())
            runs.append({"condition": condition, "seed": int(info["seed"]), "run_dir": marker.parent})
    return pd.DataFrame(runs)


def finished_runs():
    """Like completed_runs(), but stops with a clear message when there are no finished runs."""
    runs = completed_runs()
    if runs.empty:
        raise FileNotFoundError(
            f"No finished runs found in {EXPERIMENTS_DIR}. Train the conditions first (notebooks 01 to 06), "
            "or point THESIS_EXPERIMENTS_DIR to the folder that holds the run folders (see the README).")
    return runs


def per_run_table(runs):
    """Test-set results of every run on one row (AUC overall and per FST group, gaps, thresholds)."""
    rows = []
    for r in runs.itertuples():
        overall = pd.read_csv(r.run_dir / "test_metrics.csv")
        groups = pd.read_csv(r.run_dir / "test_results_by_fst_group.csv")
        o = overall.set_index("threshold_rule")
        g = groups[groups.threshold_rule == "fixed_0.5"].set_index("fst_group")["auc"]
        sel = json.loads((r.run_dir / "checkpoint_selection.json").read_text())
        rows.append({
            "condition": r.condition, "seed": r.seed,
            "auc": o.loc["fixed_0.5", "auc"],
            "auc_fst_1_2": g["FST 1-2"], "auc_fst_3_4": g["FST 3-4"], "auc_fst_5_6": g["FST 5-6"],
            "auc_gap_max_min": g.max() - g.min(),
            "auc_gap_1_2_minus_5_6": g["FST 1-2"] - g["FST 5-6"],
            "balanced_accuracy_0.5": o.loc["fixed_0.5", "balanced_accuracy"],
            "balanced_accuracy_youden": o.loc["val_youden", "balanced_accuracy"],
            "recall_youden": o.loc["val_youden", "malignant_recall"],
            "specificity_youden": o.loc["val_youden", "specificity"],
            "youden_threshold": o.loc["val_youden", "threshold"],
            "stage2_best_epoch": sel["stage2_best_epoch"],
            "val_auc": sel["stage2_best_val_auc"],
        })
    return pd.DataFrame(rows)


def seed_summary(per_run):
    """Mean and standard deviation over seeds for every column of per_run_table()."""
    metrics = [c for c in per_run.columns if c not in ("condition", "seed")]
    order = [c for c in CONDITIONS if c in set(per_run.condition)]
    grouped = per_run.groupby("condition")[metrics]
    out = (
        grouped.mean().add_suffix("_mean")
        .join(grouped.std(ddof=1).add_suffix("_sd"))
        .join(per_run.groupby("condition").size().rename("n_seeds"))
    )
    return out.loc[order]


def load_seed_predictions(runs, filename, id_column, reference="baseline"):
    """Stack the predicted probabilities of all conditions and seeds.

    Only conditions that have every seed of the reference condition are used.
    Returns (conditions, seeds, base, probs). base is the prediction file of the
    first run and is used for the labels and groups, probs has the shape
    (n_conditions, n_seeds, n_images).
    """
    conditions = [c for c in CONDITIONS if c in set(runs.condition)]
    seeds = sorted(set(runs[runs.condition == reference].seed))
    conditions = [c for c in conditions if set(runs[runs.condition == c].seed) >= set(seeds)]
    base, stacks = None, []
    for c in conditions:
        per_seed = []
        for s in seeds:
            run_dir = runs[(runs.condition == c) & (runs.seed == s)].run_dir.iloc[0]
            d = pd.read_csv(run_dir / filename)
            if base is None:
                base = d
            assert (d[id_column].values == base[id_column].values).all(), "Image order differs between runs."
            per_seed.append(d.predicted_prob.to_numpy())
        stacks.append(per_seed)
    return conditions, seeds, base, np.array(stacks)


# ---------------------------------------------------------------------------
# Bootstrap of the seed-averaged AUC
# ---------------------------------------------------------------------------
def auc_stats(y, groups, probs, idx):
    """The six AUC statistics for every condition and seed on the images in idx.

    Per model: overall AUC, AUC of the three FST groups, the largest gap between
    groups and the gap FST I-II minus FST V-VI (the order of AUC_STATS).
    probs has the shape (n_conditions, n_seeds, n_images).
    """
    out = np.empty(probs.shape[:2] + (6,))
    yi, gi = y[idx], groups[idx]
    masks = [gi == grp for grp in FST_GROUPS]
    for m in range(probs.shape[0]):
        for s in range(probs.shape[1]):
            p = probs[m, s, idx]
            ga = [roc_auc_score(yi[k], p[k]) for k in masks]
            out[m, s] = [roc_auc_score(yi, p), *ga, max(ga) - min(ga), ga[0] - ga[2]]
    return out


def bootstrap_auc(y, groups, probs, n_boot=2000, rng_seed=0):
    """Stratified bootstrap of the seed-averaged AUC statistics.

    Images are resampled with replacement within each FST group x label stratum,
    so every resample has the same composition as the original set. All
    conditions and seeds are scored on the same resample, which makes the
    comparison between conditions paired.

    Returns (point, boots): the estimates on the full set, shape (n_conditions, 6),
    and the bootstrap replicates, shape (n_boot, n_conditions, 6).
    """
    point = auc_stats(y, groups, probs, np.arange(len(y))).mean(axis=1)
    strata = [np.where((groups == grp) & (y == c))[0] for grp in FST_GROUPS for c in (0, 1)]
    rng = np.random.default_rng(rng_seed)
    boots = np.empty((n_boot, probs.shape[0], len(AUC_STATS)))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(s, len(s)) for s in strata])
        boots[b] = auc_stats(y, groups, probs, idx).mean(axis=1)
    return point, boots


def summarise_bootstrap(conditions, point, boots, reference="baseline", extra=None):
    """Turn the bootstrap replicates into two tables with 95% percentile intervals.

    estimates:   one row per condition with the value and interval of each statistic.
    differences: one row per condition except the reference, with the paired
                 difference (condition minus reference), its interval and whether
                 that interval excludes zero.
    extra holds columns that are the same for every condition (e.g. n_seeds).
    """
    ref = conditions.index(reference)
    estimates, differences = [], []
    for i, condition in enumerate(conditions):
        e, d = {"condition": condition, **(extra or {})}, {"condition": condition}
        for j, col in enumerate(AUC_STATS):
            lo, hi = np.percentile(boots[:, i, j], [2.5, 97.5])
            e[col], e[col + "_ci_low"], e[col + "_ci_high"] = point[i, j], lo, hi
            if i != ref:
                paired = boots[:, i, j] - boots[:, ref, j]
                lo, hi = np.percentile(paired, [2.5, 97.5])
                d[col + "_diff"], d[col + "_diff_ci_low"], d[col + "_diff_ci_high"] = point[i, j] - point[ref, j], lo, hi
                d[col + "_ci_excludes_0"] = bool(lo > 0 or hi < 0)
        estimates.append(e)
        if i != ref:
            differences.append(d)
    return pd.DataFrame(estimates), pd.DataFrame(differences)


def bootstrap_over_seeds(runs, n_boot=2000, rng_seed=0, reference="baseline"):
    """Bootstrap intervals on the Fitzpatrick17k test set for all finished conditions.

    Returns the two tables of summarise_bootstrap().
    """
    conditions, seeds, base, probs = load_seed_predictions(runs, "test_predictions.csv", "md5hash", reference)
    y = base.binary_label.to_numpy()
    groups = add_fst_group(base).fst_group.to_numpy()
    point, boots = bootstrap_auc(y, groups, probs, n_boot, rng_seed)
    return summarise_bootstrap(conditions, point, boots, reference, extra={"n_seeds": len(seeds)})


# ---------------------------------------------------------------------------
# Tables for reading (the csv files keep the full precision)
# ---------------------------------------------------------------------------
def format_ci_table(estimates):
    """Estimates as 'value [low, high]' strings."""
    out = estimates[["condition"]].copy()
    for c in REPORTED_STATS:
        out[c] = [f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
                  for v, lo, hi in zip(estimates[c], estimates[c + "_ci_low"], estimates[c + "_ci_high"])]
    return out


def format_diff_table(differences):
    """Paired differences as '+value [low, high]' strings; * marks an interval that excludes zero."""
    if not len(differences):
        return differences
    out = differences[["condition"]].copy()
    for c in REPORTED_STATS:
        out[c] = [f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("*" if excludes else "")
                  for d, lo, hi, excludes in zip(differences[c + "_diff"], differences[c + "_diff_ci_low"],
                                                 differences[c + "_diff_ci_high"], differences[c + "_ci_excludes_0"])]
    return out
