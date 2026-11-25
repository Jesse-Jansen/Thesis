import numpy as np
import pandas as pd

import analysis as A
import make_splits

RESIZE = 256  # images are resized to 256 x 256 before cropping (training.py)


def describe():
    rows = []

    def add(section, statistic, value):
        rows.append({"section": section, "statistic": statistic, "value": value})
        print(f"{section:12s} {statistic:52s} {value}")

    full = pd.read_csv(make_splits.METADATA)
    add("dataset", "images", len(full))
    add("dataset", "conditions", full["label"].nunique())
    for name, n in full["three_partition_label"].value_counts().items():
        add("dataset", f"images {name}", int(n))
    unlabelled = full["fitzpatrick_scale"] == -1
    add("dataset", "images without skin-type label", int(unlabelled.sum()))
    shares = A.add_fst_group(full[~unlabelled]).fst_group.value_counts(normalize=True)
    for group in A.FST_GROUPS:
        add("dataset", f"share of labelled images {group} (%)", round(100 * shares[group], 1))

    neoplasms = full["three_partition_label"].isin(["benign", "malignant"])
    add("subset", "benign or malignant images without skin-type label", int((neoplasms & unlabelled).sum()))
    subset = make_splits.select_images()
    add("subset", "images", len(subset))
    add("subset", "benign", int((subset.binary_label == 0).sum()))
    add("subset", "malignant", int((subset.binary_label == 1).sum()))
    counts = subset.fst_group.value_counts()
    for group in A.FST_GROUPS:
        add("subset", f"images {group}", int(counts[group]))
        add("subset", f"share {group} (%)", round(100 * counts[group] / len(subset), 1))

    train, val, test, _ = A.load_splits(check_images=False)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        add("split", f"{name} images", len(part))
    add("split", "train images FST V-VI", int((train.fst_group == "FST 5-6").sum()))
    for group in A.FST_GROUPS:
        for label, name in [(0, "benign"), (1, "malignant")]:
            add("split", f"test {group} {name}", int(((test.fst_group == group) & (test.binary_label == label)).sum()))

    # image sizes, only if the images are available
    paths = [A.IMAGE_DIR / f"{h}.jpg" for h in subset.md5hash]
    if all(p.is_file() for p in paths):
        from PIL import Image
        sizes = []
        for p in paths:
            with Image.open(p) as image:
                sizes.append(image.size)
        widths, heights = np.array(sizes).T
        add("images", "median width (pixels)", float(np.median(widths)))
        add("images", "median height (pixels)", float(np.median(heights)))
        add("images", f"images smaller than {RESIZE} on one side", int((np.minimum(widths, heights) < RESIZE).sum()))
    else:
        print(f"Image sizes skipped: images not found in {A.IMAGE_DIR}.")

    return pd.DataFrame(rows)


def main():
    table = describe()
    A.SUMMARY_DIR.mkdir(exist_ok=True)
    table.to_csv(A.SUMMARY_DIR / "data_description.csv", index=False)
    print("Saved", A.SUMMARY_DIR / "data_description.csv")


if __name__ == "__main__":
    main()
