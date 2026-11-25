"""Model, augmentation, two-stage training and evaluation of one run."""
import os

# needed for deterministic cuBLAS, set before CUDA starts
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy
import hashlib
import json
import math
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

import analysis as A

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HP = {
    "batch_size": 32,
    "num_workers": 0,
    "stage1": {"trainable": "head", "optimizer": "Adam", "lr": 1e-3, "max_epochs": 20, "patience": 3},
    "stage2": {"trainable": "layer4 and head", "optimizer": "Adam", "lr": 1e-4, "max_epochs": 40, "patience": None,
               "scheduler": "CosineAnnealingLR T_max=40"},
    "selection": "maximum validation AUC within each stage; stage 2 runs all epochs (no early stopping) and keeps the best-validation-AUC epoch",
    "colour_jitter": {"brightness": 0.3, "saturation": 0.3, "hue": 0.05, "probability": 0.5,
                      "contrast": 0.0, "order": "after spatial augmentation"},
    "effective_number_beta": 0.999,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# training: spatial augmentation, optional colour jitter, then tensor
spatial_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
])
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class RandomColourJitter:
    """Mild brightness, saturation and hue jitter that is applied with probability p."""

    def __init__(self, brightness=0.3, saturation=0.3, hue=0.05, p=0.5):
        self.p = p
        self.jitter = transforms.ColorJitter(brightness=brightness, saturation=saturation, hue=hue)

    def __call__(self, image):
        if torch.rand(1).item() < self.p:
            return self.jitter(image)
        return image


colour_jitter = RandomColourJitter(
    brightness=HP["colour_jitter"]["brightness"],
    saturation=HP["colour_jitter"]["saturation"],
    hue=HP["colour_jitter"]["hue"],
    p=HP["colour_jitter"]["probability"],
)


def colour_eligible(colour_mode, fitzpatrick_scale):
    """Whether an image of this skin type gets colour jitter in the given mode."""
    if colour_mode == "none":
        return False
    if colour_mode == "universal":
        return True
    if colour_mode == "targeted":
        return int(fitzpatrick_scale) in A.TARGET_FST_TYPES
    raise ValueError(f"Unknown colour mode: {colour_mode}")


class FitzpatrickDataset(Dataset):
    """Fitzpatrick17k images with their label and loss weight."""

    def __init__(self, dataframe, training=False, colour_mode="none", sample_weight=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.training = training
        self.colour_mode = colour_mode
        self.weights = np.ones(len(self.df)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        assert len(self.weights) == len(self.df)
        if not training:
            assert colour_mode == "none" and sample_weight is None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(A.IMAGE_DIR / f"{row['md5hash']}.jpg") as opened:
            image = opened.convert("RGB")
        if self.training:
            image = spatial_transform(image)
            if colour_eligible(self.colour_mode, row["fitzpatrick_scale"]):
                image = colour_jitter(image)
            image = tensor_transform(image)
        else:
            image = eval_transform(image)
        label = torch.tensor(int(row["binary_label"]), dtype=torch.float32)
        weight = torch.tensor(float(self.weights[idx]), dtype=torch.float32)
        return image, label, weight


class BinaryResNet50(nn.Module):
    """ResNet-50 pretrained on ImageNet with a small head for benign versus malignant."""

    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


def set_seed(seed):
    """Seed Python, NumPy and PyTorch and switch cuDNN to deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_epoch(model, loader, optimizer=None, desc=""):
    """One epoch. Trains when an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train() if training else model.eval()
    if training and not any(p.requires_grad for p in model.backbone.parameters()):
        # keep BatchNorm statistics of the frozen backbone fixed
        model.backbone.eval()
    total_loss, n = 0.0, 0
    labels_all, probs_all = [], []
    with torch.set_grad_enabled(training):
        for images, labels, weights in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(images)
            per_sample = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = (per_sample * weights.to(DEVICE)).mean() if training else per_sample.mean()
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
            labels_all.append(labels.cpu().numpy())
            probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
    y = np.concatenate(labels_all).astype(int)
    p = np.concatenate(probs_all)
    m = A.calculate_metrics(y, p, A.FIXED_THRESHOLD)
    return {"loss": total_loss / n, "auc": m["auc"], "accuracy": m["accuracy"],
            "malignant_recall": m["malignant_recall"], "specificity": m["specificity"],
            "malignant_f1": m["malignant_f1"]}


@torch.no_grad()
def predict(model, frame):
    """Return a copy of frame with the predicted probability of malignancy for every image."""
    loader = DataLoader(FitzpatrickDataset(frame), batch_size=HP["batch_size"], shuffle=False,
                        num_workers=HP["num_workers"], pin_memory=(DEVICE == "cuda"))
    model.eval()
    probs, labels = [], []
    for images, y, _ in tqdm(loader, desc="Predict", leave=False, dynamic_ncols=True):
        probs.append(torch.sigmoid(model(images.to(DEVICE))).cpu().numpy())
        labels.append(y.numpy().astype(int))
    out = frame.reset_index(drop=True).copy()
    assert np.array_equal(np.concatenate(labels), out.binary_label.to_numpy(dtype=int)), "Prediction order mismatch."
    out["predicted_prob"] = np.concatenate(probs)
    return out


def _log_epoch(stage, epoch, tr, va, extra=""):
    print(f"[{stage} | epoch {epoch}] train loss={tr['loss']:.4f} | val loss={va['loss']:.4f} | "
          f"val AUC={va['auc']:.4f}{extra}")


def train_two_stage(model, train_loader, val_loader, run_dir, checkpoint_path):
    """Train the model in two stages and leave the best stage-2 weights loaded."""
    # stage 1: head only
    s1 = HP["stage1"]
    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(model.head.parameters(), lr=s1["lr"])
    best1, best1_epoch, best1_state, wait, hist1 = -1.0, -1, None, 0, []
    for epoch in range(1, s1["max_epochs"] + 1):
        tr = run_epoch(model, train_loader, optimizer, desc=f"Stage 1 | epoch {epoch} | train")
        va = run_epoch(model, val_loader, desc=f"Stage 1 | epoch {epoch} | val")
        hist1.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                      **{f"val_{k}": v for k, v in va.items()}})
        pd.DataFrame(hist1).to_csv(run_dir / "stage1_training_history.csv", index=False)
        if va["auc"] > best1:
            best1, best1_epoch, best1_state, wait = va["auc"], epoch, copy.deepcopy(model.state_dict()), 0
            _log_epoch("Stage 1", epoch, tr, va, "  <- best")
        else:
            wait += 1
            _log_epoch("Stage 1", epoch, tr, va, f"  (no improvement {wait}/{s1['patience']})")
        if wait >= s1["patience"]:
            break
    model.load_state_dict(best1_state)

    # stage 2: layer4 and head
    s2 = HP["stage2"]
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone[7].parameters():  # backbone[7] is layer4 of the ResNet
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=s2["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=s2["max_epochs"])

    def keep_frozen_backbone_eval(backbone, inputs):
        # keep BatchNorm statistics of the frozen layers fixed
        for index, module in enumerate(backbone):
            if index != 7:
                module.eval()

    hook = model.backbone.register_forward_pre_hook(keep_frozen_backbone_eval)
    best2, best2_epoch, best2_state, wait, hist2 = -math.inf, None, None, 0, []
    try:
        for epoch in range(1, s2["max_epochs"] + 1):
            lr = optimizer.param_groups[0]["lr"]
            tr = run_epoch(model, train_loader, optimizer, desc=f"Stage 2 | epoch {epoch} | train")
            va = run_epoch(model, val_loader, desc=f"Stage 2 | epoch {epoch} | val")
            if not math.isfinite(va["auc"]):
                raise ValueError(f"Validation AUC is not finite at stage 2 epoch {epoch}.")
            hist2.append({"epoch": epoch, "learning_rate": lr, **{f"train_{k}": v for k, v in tr.items()},
                          **{f"val_{k}": v for k, v in va.items()}})
            pd.DataFrame(hist2).to_csv(run_dir / "stage2_training_history.csv", index=False)
            if va["auc"] > best2:
                best2, best2_epoch, best2_state, wait = va["auc"], epoch, copy.deepcopy(model.state_dict()), 0
                torch.save(best2_state, checkpoint_path)
                _log_epoch("Stage 2", epoch, tr, va, "  <- best, saved")
            else:
                wait += 1
                _log_epoch("Stage 2", epoch, tr, va, f"  (best is still epoch {best2_epoch}, val AUC={best2:.4f})")
            scheduler.step()
            if s2["patience"] is not None and wait >= s2["patience"]:
                break
    finally:
        hook.remove()
    if best2_state is None:
        raise RuntimeError("Stage 2 did not produce a checkpoint.")
    model.load_state_dict(best2_state)
    model.eval()
    return {"stage1_best_epoch": int(best1_epoch), "stage1_best_val_auc": float(best1),
            "stage2_best_epoch": int(best2_epoch), "stage2_best_val_auc": float(best2)}


def run_condition(condition, seed, overwrite=False):
    """Train and evaluate one condition with one seed."""
    spec = A.CONDITIONS[condition]
    run_dir = A.condition_dir(condition) / f"seed_{seed}"
    marker = run_dir / "run_complete.json"
    if marker.is_file() and not overwrite:
        saved = json.loads((run_dir / "config.json").read_text())
        if saved.get("stage1") != HP["stage1"] or saved.get("stage2") != HP["stage2"]:
            raise RuntimeError(
                f"{run_dir} was trained with different settings than the current training.py "
                f"(saved stage 2: {saved.get('stage2')}). Move that folder away or call "
                f"run_condition(..., overwrite=True).")
        print(f"Skipping {condition} seed {seed}: already complete ({run_dir}).")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"\n=== {condition} | seed {seed} ===\n{spec['description']}\nRun folder: {run_dir}")
    print(f"Stage 2: {HP['stage2']['max_epochs']} epochs, early stopping: "
          f"{'OFF' if HP['stage2']['patience'] is None else 'patience ' + str(HP['stage2']['patience'])}")

    set_seed(seed)
    train_df, val_df, test_df, split_hashes = A.load_splits()
    sample_weight, weight_table = A.make_sample_weights(train_df, spec["weighting"])
    if weight_table is not None:
        weight_table.to_csv(run_dir / "training_weight_table.csv", index=False)
        print(weight_table.to_string(index=False))

    config = {
        "condition": condition, "seed": seed, "description": spec["description"],
        "colour_mode": spec["colour_mode"], "weighting": spec["weighting"],
        "target_fst_types": list(A.TARGET_FST_TYPES) if spec["colour_mode"] == "targeted" else None,
        "n_colour_eligible_train_images": int(sum(colour_eligible(spec["colour_mode"], s) for s in train_df.fitzpatrick_scale)),
        "task": "binary benign versus malignant", "model": "ResNet50 IMAGENET1K_V2",
        "head": "2048-256-ReLU-Dropout(0.3)-1",
        "spatial_augmentation": "Resize 256x256; RandomResizedCrop 224 scale=(0.8,1); RandomHorizontalFlip p=0.5",
        "eval_preprocessing": "Resize 256x256; CenterCrop 224; ImageNet normalization",
        "loss": "mean(weight * BCEWithLogits) for training; unweighted BCE for validation",
        **HP,
        "thresholds": "fixed 0.5 and Youden-optimal threshold chosen on the validation set",
        "split_sha256": split_hashes, "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "image_dir": str(A.IMAGE_DIR),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "cuda": torch.version.cuda},
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    generator = torch.Generator().manual_seed(seed)
    pin = DEVICE == "cuda"
    train_loader = DataLoader(
        FitzpatrickDataset(train_df, training=True, colour_mode=spec["colour_mode"], sample_weight=sample_weight),
        batch_size=HP["batch_size"], shuffle=True, generator=generator,
        num_workers=HP["num_workers"], pin_memory=pin)
    val_loader = DataLoader(FitzpatrickDataset(val_df), batch_size=HP["batch_size"], shuffle=False,
                            num_workers=HP["num_workers"], pin_memory=pin)

    model = BinaryResNet50().to(DEVICE)
    checkpoint_path = run_dir / f"{condition}_seed{seed}_resnet50_binary.pt"
    selection = train_two_stage(model, train_loader, val_loader, run_dir, checkpoint_path)
    selection["checkpoint"] = str(checkpoint_path)
    selection["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    (run_dir / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2))

    # evaluate the saved checkpoint, not the model in memory
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    val_pred = predict(model, val_df)
    test_pred = predict(model, test_df)
    thr = A.youden_threshold(val_pred.binary_label, val_pred.predicted_prob)
    thresholds = {"fixed_0.5": A.FIXED_THRESHOLD, "val_youden": thr}
    for name, t in thresholds.items():
        test_pred[f"predicted_label_{name}"] = (test_pred.predicted_prob >= t).astype(int)
    val_pred.to_csv(run_dir / "val_predictions.csv", index=False)
    test_pred.to_csv(run_dir / "test_predictions.csv", index=False)

    overall, by_group, gaps = A.evaluate_predictions(test_pred, thresholds)
    overall.to_csv(run_dir / "test_metrics.csv", index=False)
    by_group.to_csv(run_dir / "test_results_by_fst_group.csv", index=False)
    gaps.to_csv(run_dir / "test_disparities.csv", index=False)

    # written last, marks the run as finished
    minutes = (time.time() - started) / 60
    record = {"condition": condition, "seed": seed, "checkpoint_sha256": selection["checkpoint_sha256"],
              "test_split_sha256": split_hashes["test"], "val_youden_threshold": thr,
              "test_auc": float(overall.loc[0, "auc"]), "runtime_minutes": round(minutes, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    marker.write_text(json.dumps(record, indent=2))
    print(f"\nDone: {condition} seed {seed} | test AUC={record['test_auc']:.4f} | "
          f"Youden threshold={thr:.3f} | {minutes:.1f} min")
    print(by_group[by_group.threshold_rule == "fixed_0.5"][["fst_group", "n", "auc", "balanced_accuracy"]]
          .to_string(index=False, float_format="%.4f"))
    return run_dir


def run_seeds(condition, seeds=A.DEFAULT_SEEDS, overwrite=False):
    """Run a condition for several seeds one after another and print the summary."""
    for seed in seeds:
        run_condition(condition, seed, overwrite=overwrite)
    return summarise_condition(condition)


def summarise_condition(condition):
    """Print the test results per seed and the mean (SD) over seeds for one condition."""
    runs = A.completed_runs()
    runs = runs[runs.condition == condition] if len(runs) else runs
    if not len(runs):
        print("No completed runs yet.")
        return None
    table = A.per_run_table(runs)
    cols = ["seed", "auc", "auc_fst_1_2", "auc_fst_3_4", "auc_fst_5_6", "auc_gap_1_2_minus_5_6",
            "balanced_accuracy_youden", "stage2_best_epoch"]
    print(table[cols].to_string(index=False, float_format="%.4f"))
    print("\nMean (SD) over seeds:")
    for c in cols[1:-1]:
        print(f"  {c:28s} {table[c].mean():.4f} ({table[c].std(ddof=1) if len(table) > 1 else float('nan'):.4f})")
    return table


def save_augmentation_preview(condition, n_images=8, seed=0):
    """Save a grid of FST V-VI training images with three jittered versions of each."""
    import matplotlib.pyplot as plt

    train_df, _, _, _ = A.load_splits(check_images=False)
    rows = train_df[train_df.fitzpatrick_scale.isin(A.TARGET_FST_TYPES)].sample(n_images, random_state=seed)
    torch.manual_seed(seed)
    always = transforms.ColorJitter(brightness=HP["colour_jitter"]["brightness"],
                                    saturation=HP["colour_jitter"]["saturation"],
                                    hue=HP["colour_jitter"]["hue"])
    fig, axes = plt.subplots(4, n_images, figsize=(2 * n_images, 8.4))
    for j, row in enumerate(rows.itertuples()):
        with Image.open(A.IMAGE_DIR / f"{row.md5hash}.jpg") as opened:
            image = opened.convert("RGB").resize((224, 224))
        for i in range(4):
            ax = axes[i, j]
            ax.imshow(image if i == 0 else always(image))
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                ax.set_ylabel("Original" if i == 0 else f"Jittered {i}", fontsize=11)
        axes[0, j].set_title(f"FST {row.fitzpatrick_scale}, {'malignant' if row.binary_label else 'benign'}", fontsize=9)
    fig.suptitle("Colour jitter: brightness and saturation x[0.7, 1.3], hue +/-0.05 (applied with p=0.5 in training)",
                 fontsize=11)
    fig.tight_layout()
    path = A.condition_dir(condition) / "augmentation_preview.png"
    fig.savefig(path, dpi=150)
    plt.show()
    print("Saved", path)
    return path
