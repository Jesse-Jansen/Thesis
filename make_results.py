"""Build the results figure and the LaTeX tables of the thesis from the csv files in summary/.

    python make_results.py [summary_folder] [output_folder]

Without arguments it reads summary/ and writes to figures/. Run 00_aggregate_results.ipynb and
07_ddi_external_evaluation.ipynb first. The two tables of the threshold-based
analysis are only written when threshold_analysis.py has been run as well.

The table bodies are generated here. The captions in the thesis were extended
by hand afterwards.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import pandas as pd

import analysis as A

REQUIRED = [
    "bootstrap_auc.csv", "bootstrap_diff_vs_baseline.csv",
    "ddi_bootstrap_auc.csv", "ddi_bootstrap_diff_vs_baseline.csv",
    "ddi_sensitivity_bootstrap_auc.csv", "ddi_sensitivity_bootstrap_diff_vs_baseline.csv",
]

NAMES = {
    "baseline": "Baseline",
    "universal_colour": "Universal colour",
    "targeted_colour": "Targeted colour",
    "reweighting_effective_number": "Effective-number",
    "reweighting_kamiran": "Kamiran",
    "combined": "Combined",
}

# Column in the csv files and its header in the LaTeX tables.
COLS = [
    ("auc", "Overall"),
    ("auc_fst_1_2", "FST I--II"),
    ("auc_fst_3_4", "FST III--IV"),
    ("auc_fst_5_6", "FST V--VI"),
    ("auc_gap_1_2_minus_5_6", "Gap"),
]
N_IMAGES = {"Fitzpatrick17k test set": 648, "DDI": 656}


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------
def tex_number(x, signed=False):
    """Three decimals, with a proper minus sign (and plus sign if signed) in math mode."""
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    return s.replace("-", "$-$").replace("+", "$+$") if signed or x < 0 else s


def estimate_rows(estimates, differences):
    """One table row per condition: value [low, high]. A star marks a reliable difference from the baseline."""
    differences = differences.set_index("condition")
    rows = []
    for r in estimates.itertuples():
        cells = [NAMES[r.condition]]
        for col, _ in COLS:
            signed = col.startswith("auc_gap")
            star = "*" if r.condition != "baseline" and bool(differences.loc[r.condition, col + "_ci_excludes_0"]) else ""
            value = tex_number(getattr(r, col), signed)
            low = tex_number(getattr(r, col + "_ci_low"), signed)
            high = tex_number(getattr(r, col + "_ci_high"), signed)
            cells.append(f"{value}{star} [{low}, {high}]")
        rows.append(" & ".join(cells) + r" \\")
    return rows


def difference_rows(differences):
    """One table row per condition: paired difference from the baseline [low, high]."""
    rows = []
    for r in differences.itertuples():
        cells = [NAMES[r.condition]]
        for col, _ in COLS:
            star = "*" if getattr(r, col + "_ci_excludes_0") else ""
            value = tex_number(getattr(r, col + "_diff"), True)
            low = tex_number(getattr(r, col + "_diff_ci_low"), True)
            high = tex_number(getattr(r, col + "_diff_ci_high"), True)
            cells.append(f"{value}{star} [{low}, {high}]")
        rows.append(" & ".join(cells) + r" \\")
    return rows


def latex_table(label, caption, header, blocks, note=None, column_spec="@{}llllll@{}"):
    """A booktabs table. blocks is a list of (subheading, rows); a subheading of None is skipped."""
    n_columns = header.count("&") + 1
    body = []
    for i, (title, rows) in enumerate(blocks):
        if i:
            body.append(r"\addlinespace")
        if title is not None:
            body.append(r"\multicolumn{" + str(n_columns) + r"}{@{}l}{\textit{" + title + r"}} \\")
        body += rows
    lines = [
        r"\begin{table*}[t]",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{" + column_spec + "}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if note:
        lines.append(r"\par\vspace{2pt}\parbox{0.92\textwidth}{\footnotesize " + note + "}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def write_auc_tables(summary_dir, out_dir):
    """Tables with the AUC per condition, the paired differences and the sensitivity analysis."""
    estimates = {
        "Fitzpatrick17k test set": pd.read_csv(summary_dir / "bootstrap_auc.csv"),
        "DDI": pd.read_csv(summary_dir / "ddi_bootstrap_auc.csv"),
    }
    differences = {
        "Fitzpatrick17k test set": pd.read_csv(summary_dir / "bootstrap_diff_vs_baseline.csv"),
        "DDI": pd.read_csv(summary_dir / "ddi_bootstrap_diff_vs_baseline.csv"),
    }
    sens_estimates = pd.read_csv(summary_dir / "ddi_sensitivity_bootstrap_auc.csv")
    sens_differences = pd.read_csv(summary_dir / "ddi_sensitivity_bootstrap_diff_vs_baseline.csv")
    header = "Condition & " + " & ".join(name for _, name in COLS) + r" \\"

    table_auc = latex_table(
        "tab:auc", "ROC-AUC per condition, averaged over three seeds, with 95\\% bootstrap intervals.", header,
        [(f"{name} ($n = {N_IMAGES[name]}$)", estimate_rows(estimates[name], differences[name])) for name in estimates],
        "Gap = AUC of FST I--II minus AUC of FST V--VI. * The 95\\% interval of the paired difference "
        "from the baseline excludes zero (Table~\\ref{tab:diff}).")
    table_diff = latex_table(
        "tab:diff", "Paired difference in ROC-AUC from the baseline (condition minus baseline), with 95\\% bootstrap intervals.",
        header, [(name, difference_rows(differences[name])) for name in differences],
        "* The interval excludes zero.")
    table_sens = latex_table(
        "tab:sens", "Sensitivity analysis on DDI without the 89 non-neoplastic images ($n = 567$).", header,
        [("ROC-AUC", estimate_rows(sens_estimates, sens_differences)),
         ("Paired difference from the baseline", difference_rows(sens_differences))],
        "* The 95\\% interval of the paired difference from the baseline excludes zero.")

    (out_dir / "table_auc.tex").write_text(table_auc, encoding="utf-8")
    (out_dir / "table_diff.tex").write_text(table_diff, encoding="utf-8")
    (out_dir / "table_sens.tex").write_text(table_sens, encoding="utf-8")
    return differences


def write_threshold_tables(summary_dir, out_dir):
    """Tables of the threshold-based analysis: sensitivity / specificity and errors per DDI diagnosis."""
    results = pd.read_csv(summary_dir / "threshold_sens_spec.csv")
    display_names = dict(zip(A.CONDITIONS, NAMES.values()))
    groups = ["Overall", *A.FST_GROUPS]

    def cell(split, condition, group):
        parts = []
        for metric in ["Sensitivity", "Specificity"]:
            r = results[(results.split == split) & (results.condition == condition)
                        & (results.group == group) & (results.metric == metric)].iloc[0]
            reliable = condition != "baseline" and (r.diff_ci_low > 0 or r.diff_ci_high < 0)
            parts.append(f"{r.estimate:.3f}" + ("*" if reliable else ""))
        return " / ".join(parts)

    blocks = []
    for split, title in [("test", "Fitzpatrick17k test set"), ("ddi", "DDI")]:
        rows = [display_names[c] + " & " + " & ".join(cell(split, c, g) for g in groups) + r" \\" for c in A.CONDITIONS]
        blocks.append((f"{title} ($n = {N_IMAGES[title]}$)", rows))
    table_threshold = latex_table(
        "tab:thresh",
        "Sensitivity / specificity per condition at a threshold chosen on the Fitzpatrick17k validation set "
        "(Youden's index), averaged over three seeds, on the Fitzpatrick17k test set (top) and DDI (bottom). "
        "Columns give the values over all test images (Overall) and per FST group.",
        r"Condition & Overall & FST I--II & FST III--IV & FST V--VI \\", blocks,
        "* The 95\\% interval of the paired difference from the baseline excludes zero.",
        column_spec="@{}lllll@{}")
    (out_dir / "table_threshold.tex").write_text(table_threshold, encoding="utf-8")

    # Errors per diagnosis, for diagnoses with at least ten DDI images.
    errors = pd.read_csv(summary_dir / "threshold_ddi_errors_by_diagnosis.csv")
    errors = errors[errors.n >= 10]
    baseline = errors[errors.condition == "baseline"].set_index(["disease", "binary_label"])
    universal = errors[errors.condition == "universal_colour"].set_index(["disease", "binary_label"])
    table = (
        baseline[["n", "error_rate"]].rename(columns={"error_rate": "baseline"})
        .join(universal["error_rate"].rename("universal"))
        .reset_index()
        .sort_values(["binary_label", "baseline"], ascending=[True, False])
    )
    special_names = {
        "seborrheic-keratosis-irritated": "Seborrheic keratosis (irritated)",
        "melanocytic-nevi": "Melanocytic nevus",
    }
    rows = []
    for r in table.itertuples():
        name = special_names.get(r.disease, r.disease.replace("-", " ").capitalize())
        label = "Malignant" if r.binary_label == 1 else "Benign"
        rows.append(f"{name} & {label} & {int(r.n)} & {r.baseline:.2f} & {r.universal:.2f} \\\\")
    table_errors = latex_table(
        "tab:diag",
        "Error rate per DDI diagnosis (diagnoses with at least ten images) at the validation-chosen threshold, "
        "averaged over three seeds. For benign diagnoses the error rate is the share of images classified as "
        "malignant; for malignant diagnoses, the share classified as benign.",
        r"Diagnosis & Label & $n$ & Baseline & Universal colour \\", [(None, rows)],
        column_spec="@{}llrrr@{}")
    (out_dir / "table_ddi_errors.tex").write_text(table_errors, encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure: paired differences from the baseline (Figure A2 of the thesis)
# ---------------------------------------------------------------------------
def plot_differences(differences, out_dir):
    """Forest plot with one row per test set and one panel per statistic.

    A filled marker means that the 95% interval of the difference excludes zero.
    """
    ink, muted, grid, blue = "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": muted, "axes.linewidth": 0.6,
        "xtick.color": muted, "ytick.color": ink, "text.color": ink, "axes.labelcolor": ink,
    })
    panels = [("auc", "Overall"), ("auc_fst_1_2", "FST I–II"), ("auc_fst_3_4", "FST III–IV"),
              ("auc_fst_5_6", "FST V–VI"), ("auc_gap_1_2_minus_5_6", "Gap (I–II minus V–VI)")]

    fig, axes = plt.subplots(2, 5, figsize=(7.16, 3.4), sharey=True, sharex="row")
    for i, (dataset, table) in enumerate(differences.items()):
        for j, (col, title) in enumerate(panels):
            ax = axes[i, j]
            ax.axvline(0, color=muted, lw=0.8, zorder=1)
            for k, r in enumerate(table.itertuples()):
                y = len(table) - 1 - k  # first condition at the top
                value = getattr(r, col + "_diff")
                low, high = getattr(r, col + "_diff_ci_low"), getattr(r, col + "_diff_ci_high")
                excludes_zero = getattr(r, col + "_ci_excludes_0")
                ax.plot([low, high], [y, y], color=blue, lw=1.6, solid_capstyle="round", zorder=2)
                ax.plot(value, y, "o", ms=5.5, mfc=blue if excludes_zero else "white", mec=blue, mew=1.4, zorder=3)
            ax.set_yticks(range(len(table)))
            ax.set_yticklabels([NAMES[c] for c in table.condition][::-1])
            ax.set_ylim(-0.6, len(table) - 0.4)
            ax.grid(axis="x", color=grid, lw=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", length=2, labelsize=7)
            ax.set_title(title, fontsize=8, pad=4)
        # The differences on DDI are larger, so the two rows get their own x-axis range.
        limit = 0.05 if i == 0 else 0.15
        axes[i, 0].set_xlim(-limit, limit)
        for ax in axes[i]:
            ax.set_xticks([-0.1, 0, 0.1] if i else [-0.04, 0, 0.04])
        fig.text(0.008, 0.955 if i == 0 else 0.505, dataset, fontsize=8, fontweight="bold", ha="left", va="center")

    fig.text(0.19, 0.032, "Difference in AUC from the baseline (95% interval)", fontsize=8, ha="left", va="center")
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=5.5, mfc=blue, mec=blue, mew=1.4, label="Interval excludes zero"),
        plt.Line2D([], [], marker="o", ls="", ms=5.5, mfc="white", mec=blue, mew=1.4, label="Interval includes zero"),
    ]
    fig.legend(handles=handles, loc="lower right", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(1.0, -0.012), handletextpad=0.2, columnspacing=0.8)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.86, bottom=0.15, wspace=0.14, hspace=0.62)
    fig.savefig(out_dir / "results_differences.pdf")
    fig.savefig(out_dir / "results_differences.png", dpi=200)


def main():
    summary_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else A.SUMMARY_DIR
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else A.FIGURES_DIR
    missing = [name for name in REQUIRED if not (summary_dir / name).is_file()]
    if missing:
        sys.exit(f"Missing in {summary_dir}: {missing}. "
                 "Run 00_aggregate_results.ipynb and 07_ddi_external_evaluation.ipynb first.")
    out_dir.mkdir(parents=True, exist_ok=True)

    differences = write_auc_tables(summary_dir, out_dir)
    plot_differences(differences, out_dir)
    if (summary_dir / "threshold_sens_spec.csv").is_file():
        write_threshold_tables(summary_dir, out_dir)
    else:
        print("No threshold tables: run threshold_analysis.py to create them.")
    print("Wrote figure and tables to", out_dir)


if __name__ == "__main__":
    main()
