from pathlib import Path

import numpy as np
import pandas as pd

import analysis as A

DDI_DIR = A.path_setting("DDI_DIR", A.PROJECT_DIR.parent / "ddi_thesis_data")
DDI_IMAGE_DIR = DDI_DIR / "images"
DDI_METADATA = DDI_DIR / "ddi_metadata.csv"

SKIN_TONE_TO_GROUP = {12: "FST 1-2", 34: "FST 3-4", 56: "FST 5-6"}

# Non-neoplastic DDI diagnoses, left out in the sensitivity analysis.
NON_NEOPLASTIC_DISEASES = {
    "verruca-vulgaris", "wart", "molluscum-contagiosum", "condyloma-accuminatum",
    "eczema-spongiotic-dermatitis", "abrasions-ulcerations-and-physical-injuries",
    "hyperpigmentation", "onychomycosis", "tinea-pedis", "coccidioidomycosis",
    "scar", "keloid", "hematoma", "abscess", "folliculitis", "acne-cystic",
    "morphea", "dermatomyositis", "graft-vs-host-disease", "foreign-body-granuloma",
    "lymphocytic-infiltrations", "reactive-lymphoid-hyperplasia",
    "focal-acral-hyperkeratosis",
}


def load_metadata(metadata_path=DDI_METADATA, image_dir=DDI_IMAGE_DIR, check_images=True):
    """Read the DDI metadata and add binary_label, fst_group and non_neoplastic."""
    df = pd.read_csv(metadata_path)
    assert len(df) == 656 and df.DDI_file.is_unique, "Expected the 656-image DDI metadata file."
    df["binary_label"] = df["malignant"].astype(int)
    df["fst_group"] = df["skin_tone"].map(SKIN_TONE_TO_GROUP)
    assert df.fst_group.notna().all(), "Unexpected skin_tone value."
    unknown = NON_NEOPLASTIC_DISEASES - set(df.disease)
    assert not unknown, f"Not in DDI metadata: {unknown}"
    df["non_neoplastic"] = df.disease.isin(NON_NEOPLASTIC_DISEASES)
    assert not (df.non_neoplastic & (df.binary_label == 1)).any(), "A malignant diagnosis is on the exclusion list."
    if check_images:
        missing = [f for f in df.DDI_file if not (Path(image_dir) / f).is_file()]
        assert not missing, f"Missing DDI images: {missing[:5]}"
    return df[["DDI_ID", "DDI_file", "skin_tone", "fst_group", "disease", "binary_label", "non_neoplastic"]]


def describe(df):
    """Number of images, malignant images and non-neoplastic images per FST group."""
    t = df.groupby("fst_group").agg(n=("binary_label", "size"), n_malignant=("binary_label", "sum"),
                                    n_non_neoplastic=("non_neoplastic", "sum")).reindex(A.FST_GROUPS)
    t.loc["All"] = t.sum()
    t["n_after_exclusion"] = t.n - t.n_non_neoplastic
    return t.astype(int)


def predict_run(run_dir, df, overwrite=False):
    """Score one finished run on all DDI images and save ddi_predictions.csv in its folder."""
    run_dir = Path(run_dir)
    out_path = run_dir / "ddi_predictions.csv"
    if out_path.is_file() and not overwrite:
        return out_path

    # imported here so the analysis also runs without PyTorch
    import json
    import hashlib
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    import training as T

    sel = json.loads((run_dir / "checkpoint_selection.json").read_text())
    ckpt = run_dir / Path(sel["checkpoint"].replace("\\", "/")).name
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == sel["checkpoint_sha256"], f"Checkpoint changed: {ckpt}"

    class DDIDataset(Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            with Image.open(DDI_IMAGE_DIR / df.DDI_file.iloc[i]) as opened:
                return T.eval_transform(opened.convert("RGB"))

    model = T.BinaryResNet50().to(T.DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
    model.eval()
    probs = []
    with torch.no_grad():
        for images in DataLoader(DDIDataset(), batch_size=T.HP["batch_size"], shuffle=False, num_workers=0):
            probs.append(torch.sigmoid(model(images.to(T.DEVICE))).cpu().numpy())
    out = df.copy()
    out["predicted_prob"] = np.concatenate(probs)
    out.to_csv(out_path, index=False)
    return out_path


def predict_all(df, overwrite=False):
    """Run predict_run() for every finished run."""
    runs = A.finished_runs()
    for r in runs.itertuples():
        path = predict_run(r.run_dir, df, overwrite=overwrite)
        print(f"{r.condition:32s} seed {r.seed}: {path.name}")
    return runs


def per_run_table(runs, exclude_non_neoplastic=False):
    """AUC statistics on DDI for every single run."""
    rows = []
    for r in runs.itertuples():
        d = pd.read_csv(r.run_dir / "ddi_predictions.csv")
        if exclude_non_neoplastic:
            d = d[~d.non_neoplastic].reset_index(drop=True)
        stats = A.auc_stats(d.binary_label.to_numpy(), d.fst_group.to_numpy(),
                            d.predicted_prob.to_numpy()[None, None, :], np.arange(len(d)))[0, 0]
        rows.append({"condition": r.condition, "seed": r.seed, "n": len(d), **dict(zip(A.AUC_STATS, stats))})
    return pd.DataFrame(rows)


def bootstrap(runs, exclude_non_neoplastic=False, n_boot=2000, rng_seed=0, reference="baseline"):
    """Bootstrap intervals on DDI, with the same procedure as for the internal test set."""
    conditions, seeds, base, probs = A.load_seed_predictions(runs, "ddi_predictions.csv", "DDI_file", reference)
    keep = ~base.non_neoplastic.to_numpy() if exclude_non_neoplastic else np.ones(len(base), bool)
    y, groups, probs = base.binary_label.to_numpy()[keep], base.fst_group.to_numpy()[keep], probs[:, :, keep]
    point, boots = A.bootstrap_auc(y, groups, probs, n_boot, rng_seed)
    return A.summarise_bootstrap(conditions, point, boots, reference,
                                 extra={"n_seeds": len(seeds), "n_images": int(keep.sum())})
