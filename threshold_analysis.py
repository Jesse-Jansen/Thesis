"""Sensitivity and specificity per FST group at the validation-chosen threshold."""
import numpy as np
import pandas as pd

import analysis as A

N_BOOT = 2000
RNG_SEED = 0

SPLITS = {
    "test": ("test_predictions.csv", "md5hash"),
    "ddi": ("ddi_predictions.csv", "DDI_file"),
}


def load_binary_predictions(runs, split):
    """Predicted labels (0 or 1) of all conditions and seeds at their validation threshold."""
    filename, id_column = SPLITS[split]
    conditions, seeds, base, probs = A.load_seed_predictions(runs, filename, id_column)
    predictions = np.empty_like(probs)
    for i, condition in enumerate(conditions):
        for j, seed in enumerate(seeds):
            run_dir = runs[(runs.condition == condition) & (runs.seed == seed)].run_dir.iloc[0]
            val = pd.read_csv(run_dir / "val_predictions.csv")
            threshold = A.youden_threshold(val.binary_label, val.predicted_prob)
            predictions[i, j] = probs[i, j] >= threshold
    return conditions, base, predictions


def sens_spec(y, masks, predictions, idx):
    """Seed-averaged sensitivity and specificity on the images in idx."""
    yi, pi = y[idx], predictions[:, :, idx]
    out = np.empty((predictions.shape[0], len(masks), 2))
    for g, mask in enumerate(masks.values()):
        in_group = mask[idx]
        malignant, benign = in_group & (yi == 1), in_group & (yi == 0)
        out[:, g, 0] = pi[:, :, malignant].mean(axis=2).mean(axis=1)
        out[:, g, 1] = 1 - pi[:, :, benign].mean(axis=2).mean(axis=1)
    return out


def analyse_split(runs, split):
    """Sensitivity and specificity with bootstrap intervals for one test set."""
    conditions, base, predictions = load_binary_predictions(runs, split)
    y = base.binary_label.to_numpy().astype(int)
    groups = base.fst_group.to_numpy()
    masks = {"Overall": np.ones(len(y), bool), **{g: groups == g for g in A.FST_GROUPS}}

    point = sens_spec(y, masks, predictions, np.arange(len(y)))
    strata = [np.where((groups == g) & (y == c))[0] for g in A.FST_GROUPS for c in (0, 1)]
    rng = np.random.default_rng(RNG_SEED)
    boots = np.empty((N_BOOT, *point.shape))
    for b in range(N_BOOT):
        idx = np.concatenate([rng.choice(s, len(s)) for s in strata])
        boots[b] = sens_spec(y, masks, predictions, idx)

    ref = conditions.index("baseline")
    rows = []
    for i, condition in enumerate(conditions):
        for g, group in enumerate(masks):
            for m, metric in enumerate(["Sensitivity", "Specificity"]):
                replicates = boots[:, i, g, m]
                paired = replicates - boots[:, ref, g, m]
                rows.append({
                    "split": split, "condition": condition, "group": group, "metric": metric,
                    "estimate": point[i, g, m],
                    "ci_low": np.quantile(replicates, 0.025), "ci_high": np.quantile(replicates, 0.975),
                    "diff_vs_baseline": point[i, g, m] - point[ref, g, m],
                    "diff_ci_low": np.quantile(paired, 0.025), "diff_ci_high": np.quantile(paired, 0.975),
                })
    return pd.DataFrame(rows), conditions, base, predictions


def errors_by_diagnosis(conditions, base, predictions, show=("baseline", "universal_colour")):
    """Share of misclassified DDI images per diagnosis, averaged over seeds."""
    y = base.binary_label.to_numpy().astype(int)
    tables = []
    for condition in show:
        wrong = (predictions[conditions.index(condition)] != y).mean(axis=0)
        table = (
            base.assign(error=wrong)
            .groupby(["disease", "binary_label"])
            .agg(n=("error", "size"), error_rate=("error", "mean"))
            .reset_index()
        )
        table["condition"] = condition
        tables.append(table)
    return pd.concat(tables).sort_values(["condition", "n"], ascending=[True, False], kind="stable")


def main():
    summary_dir = A.SUMMARY_DIR
    summary_dir.mkdir(exist_ok=True)
    runs = A.finished_runs()

    results = []
    for split in SPLITS:
        table, conditions, base, predictions = analyse_split(runs, split)
        results.append(table)
        if split == "ddi":
            errors_by_diagnosis(conditions, base, predictions).to_csv(
                summary_dir / "threshold_ddi_errors_by_diagnosis.csv", index=False)
    results = pd.concat(results)
    results.to_csv(summary_dir / "threshold_sens_spec.csv", index=False)
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
