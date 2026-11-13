"""Stricter check of the one reliable effect: universal colour augmentation on DDI.

Many comparisons were made in the thesis, so some 95% intervals can exclude zero
by chance. This script repeats the paired bootstrap for universal colour versus
the baseline with more resamples and also reports 99% intervals, for the AUC of
FST V-VI and for the gap between FST I-II and FST V-VI. It does this for all DDI
images and for the sensitivity analysis without the non-neoplastic images.

    python stricter_interval_check.py

Needs the DDI predictions of every run (07_ddi_external_evaluation.ipynb).
"""
import numpy as np

import analysis as A

CONDITION = "universal_colour"
STATISTICS = {"auc_fst_5_6": "FST V-VI AUC", "auc_gap_1_2_minus_5_6": "gap I-II minus V-VI"}
N_BOOT = 5000
RNG_SEED = 1


def check(base, probs, conditions, keep, label):
    """Print the difference from the baseline with its 95% and 99% interval."""
    y = base.binary_label.to_numpy()[keep]
    groups = base.fst_group.to_numpy()[keep]
    point, boots = A.bootstrap_auc(y, groups, probs[:, :, keep], N_BOOT, RNG_SEED)
    i, ref = conditions.index(CONDITION), conditions.index("baseline")
    for column, name in STATISTICS.items():
        j = A.AUC_STATS.index(column)
        paired = boots[:, i, j] - boots[:, ref, j]
        lo95, hi95, lo99, hi99 = np.percentile(paired, [2.5, 97.5, 0.5, 99.5])
        print(f"{label:22s} {name:20s} diff {point[i, j] - point[ref, j]:+.3f}  "
              f"95% [{lo95:+.3f}, {hi95:+.3f}]  99% [{lo99:+.3f}, {hi99:+.3f}]")


def main():
    runs = A.finished_runs()
    conditions, _, base, probs = A.load_seed_predictions(runs, "ddi_predictions.csv", "DDI_file")
    check(base, probs, conditions, np.ones(len(base), bool), "all 656 images")
    check(base, probs, conditions, ~base.non_neoplastic.to_numpy(), "without non-neoplastic")


if __name__ == "__main__":
    main()
