"""Create the train, validation and test split used in the thesis.

Keeps the Fitzpatrick17k images that are labelled benign or malignant and have
a Fitzpatrick skin type, and splits them 70/15/15, stratified by label and FST
group. The three csv files are written to <project>/binary_experiment_splits.

    python make_splits.py

The input is fitzpatrick17k.csv from https://github.com/mattgroh/fitzpatrick17k,
placed in the project folder. Running this script again gives the same files.
analysis.load_splits() checks their SHA-256, so a different split cannot be used
by accident.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

import analysis as A

SEED = 42
METADATA = A.PROJECT_DIR / "fitzpatrick17k.csv"


def select_images(metadata_path=METADATA):
    """Benign and malignant images with a known skin type, plus binary_label and fst_group."""
    df = pd.read_csv(metadata_path)
    # three_partition_label is benign, malignant or non-neoplastic. A fitzpatrick_scale
    # of -1 means that the annotators could not assign a skin type.
    df = df[
        df["three_partition_label"].isin(["benign", "malignant"])
        & (df["fitzpatrick_scale"] != -1)
    ].copy()
    df["binary_label"] = df["three_partition_label"].map({"benign": 0, "malignant": 1})
    return A.add_fst_group(df)


def split(df, seed=SEED):
    """70/15/15 split, stratified by the combination of label and FST group."""
    train_df, rest = train_test_split(
        df, test_size=0.30, random_state=seed,
        stratify=df["binary_label"].astype(str) + "_" + df["fst_group"],
    )
    val_df, test_df = train_test_split(
        rest, test_size=0.50, random_state=seed,
        stratify=rest["binary_label"].astype(str) + "_" + rest["fst_group"],
    )
    return train_df, val_df, test_df


def main():
    df = select_images()
    print(f"{len(df)} images selected")
    print(pd.crosstab(df["fst_group"], df["binary_label"]).rename(columns={0: "benign", 1: "malignant"}))

    A.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in zip(["train", "val", "test"], split(df)):
        # Windows line endings on every platform, so the files (and their hashes)
        # do not depend on the operating system.
        part.to_csv(A.SPLIT_DIR / f"{name}.csv", index=False, lineterminator="\r\n")
        print(f"Saved {name}: {len(part)} images")


if __name__ == "__main__":
    main()
