import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis as A

SPLITS = {
    "test": ("test_predictions.csv", "md5hash"),
    "ddi": ("ddi_predictions.csv", "DDI_file"),
}
GROUP_TITLES = {"FST 1-2": "FST I–II", "FST 3-4": "FST III–IV", "FST 5-6": "FST V–VI"}
CONDITION_TITLES = {"baseline": "Baseline", "universal_colour": "Universal colour"}
CLASSES = ["Benign", "Malignant"]


def confusion_per_group(runs, split):

   
    filename, id_column = SPLITS[split]
    conditions, seeds, base, probs = A.load_seed_predictions(runs, filename, id_column)
    y = base.binary_label.to_numpy().astype(int)
    groups = base.fst_group.to_numpy()

    rows = []
    for i, condition in enumerate(conditions):
        predictions = []
        for j, seed in enumerate(seeds):
            run_dir = runs[(runs.condition == condition) & (runs.seed == seed)].run_dir.iloc[0]
            val = pd.read_csv(run_dir / "val_predictions.csv")
            threshold = A.youden_threshold(val.binary_label, val.predicted_prob)
            predictions.append((probs[i, j] >= threshold).astype(int))
        predictions = np.array(predictions)

        for group in A.FST_GROUPS:
            mask = groups == group
            yg, pg = y[mask], predictions[:, mask]
            counts = {
                "tn": ((yg == 0) & (pg == 0)).sum(axis=1).mean(),
                "fp": ((yg == 0) & (pg == 1)).sum(axis=1).mean(),
                "fn": ((yg == 1) & (pg == 0)).sum(axis=1).mean(),
                "tp": ((yg == 1) & (pg == 1)).sum(axis=1).mean(),
            }
            rows.append({
                "split": split, "condition": condition, "fst_group": group,
                "n_benign": int((yg == 0).sum()), "n_malignant": int((yg == 1).sum()),
                **counts,
                "sensitivity": counts["tp"] / (counts["tp"] + counts["fn"]),
                "specificity": counts["tn"] / (counts["tn"] + counts["fp"]),
            })
    return pd.DataFrame(rows)


def draw_matrix(ax, row, show_ylabel=True, ylabel="True"):
    """One confusion matrix."""
    counts = np.array([[row.tn, row.fp], [row.fn, row.tp]])
    rates = counts / counts.sum(axis=1, keepdims=True)
    ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for r in range(2):
        for c in range(2):
            colour = "white" if rates[r, c] > 0.6 else "#0b0b0b"
            ax.text(c, r, f"{100 * rates[r, c]:.0f}%\n({counts[r, c]:.1f})",
                    ha="center", va="center", fontsize=7.5, color=colour)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(CLASSES, fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(CLASSES if show_ylabel else ["", ""], fontsize=7)
    ax.set_xlabel("Predicted", fontsize=7)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def plot(table, split, conditions, path_stem):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(len(conditions), 3, figsize=(7.0, 2.35 * len(conditions)), squeeze=False)
    for i, condition in enumerate(conditions):
        for j, group in enumerate(A.FST_GROUPS):
            row = table[(table.split == split) & (table.condition == condition) & (table.fst_group == group)].iloc[0]
            ax = axes[i, j]
            ylabel = "True" if len(conditions) == 1 else f"{CONDITION_TITLES.get(condition, condition)}\n\nTrue"
            draw_matrix(ax, row, show_ylabel=(j == 0), ylabel=ylabel)
            ax.set_title(f"{GROUP_TITLES[group]} (n = {row.n_benign + row.n_malignant})", fontsize=8, pad=4)
    fig.tight_layout()
    save_figure(fig, path_stem.with_suffix(".pdf"))
    save_figure(fig, path_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def save_figure(fig, path, **kwargs):
    # via a temporary file: on Windows a figure open in a viewer cannot be overwritten
    tmp = path.with_name(path.stem + "_tmp" + path.suffix)
    fig.savefig(tmp, **kwargs)
    try:
        os.replace(tmp, path)
    except OSError as error:
        raise OSError(f"Could not replace {path.name}; close any program that has it open "
                      f"and run the script again. The new figure is in {tmp.name}.") from error


def main():
    runs = A.finished_runs()
    tables = [confusion_per_group(runs, "test")]
    has_ddi = all((r.run_dir / "ddi_predictions.csv").is_file() for r in runs.itertuples())
    if has_ddi:
        tables.append(confusion_per_group(runs, "ddi"))
    else:
        print("No DDI predictions found: only the Fitzpatrick17k test set is used.")
    table = pd.concat(tables, ignore_index=True)

    A.SUMMARY_DIR.mkdir(exist_ok=True)
    A.FIGURES_DIR.mkdir(exist_ok=True)
    table.to_csv(A.SUMMARY_DIR / "confusion_matrices.csv", index=False)
    plot(table, "test", ["baseline"], A.FIGURES_DIR / "confusion_matrices_fitz")
    if has_ddi:
        plot(table, "ddi", ["baseline", "universal_colour"], A.FIGURES_DIR / "confusion_matrices_ddi")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
