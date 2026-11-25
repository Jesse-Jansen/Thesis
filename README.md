# Data augmentation and reweighting for skin-tone bias in dermatological image classification

Code for the bachelor thesis *Evaluating Data Augmentation and Reweighting for Reducing
Skin-Tone Bias in Clinical Dermatological Image Classification* (Jesse Jansen, BSc Cognitive
Science & Artificial Intelligence, Tilburg University, 2026).

A ResNet-50 is trained on Fitzpatrick17k to classify skin lesions as benign or malignant, under
six conditions. Every condition is trained with three seeds and evaluated per Fitzpatrick skin
type (FST) group, on an internal test set and on the external DDI dataset.

| Folder | Colour augmentation | Loss weights |
|---|---|---|
| `01_baseline` | none | none |
| `02_universal_colour` | mild jitter, all training images | none |
| `03_targeted_colour` | mild jitter, FST V–VI training images only | none |
| `04_reweighting_effective_number` | none | effective number of samples per FST group (Cui et al., 2019), β = 0.999 |
| `05_reweighting_kamiran` | none | reweighing of Kamiran & Calders (2012), FST type × label |
| `06_combined_targeted_colour_effective_number` | mild jitter, FST V–VI only | effective number per FST group |

**Mild jitter:** brightness and saturation × [0.7, 1.3], hue ±0.05 (about ±18°), applied with
probability 0.5 after the spatial augmentation.

**The same in every run:** ResNet-50 (ImageNet weights) with a 2048-256-1 head. Stage 1 trains
the head (Adam, learning rate 1e-3, at most 20 epochs, patience 3). Stage 2 trains layer4 and the
head (Adam, learning rate 1e-4, cosine annealing, 40 epochs, no early stopping). The checkpoint
is the epoch with the highest validation AUC. Seeds 42, 43 and 44, deterministic cuDNN.

## Files

| File | What it does |
|---|---|
| `analysis.py` | Paths, conditions, split checks, sample weights, metrics and the bootstrap. |
| `training.py` | Model, augmentation and training of one run. |
| `ddi.py` | Predictions and analysis on DDI. |
| `make_splits.py` | 70/15/15 split of the Fitzpatrick17k metadata. |
| `01_…` to `06_…` | One notebook per condition, trains the three seeds. |
| `00_aggregate_results.ipynb` | Result tables for the Fitzpatrick17k test set. |
| `07_ddi_external_evaluation.ipynb` | Result tables for DDI. |
| `threshold_analysis.py` | Sensitivity and specificity per FST group (Table A5, A6). |
| `stricter_interval_check.py` | 99% intervals for universal colour augmentation on DDI. |
| `confusion_matrices.py` | Confusion matrices per FST group (Fig. A3, A4). |
| `describe_data.py` | Dataset statistics reported in the thesis. |
| `make_results.py` | LaTeX tables and Fig. A2 from `summary/`. |
| `summary/`, `figures/` | Results and figures. Fig. A1 is made by `training.save_augmentation_preview()`. |

## Data

The images are not part of this repository.

- **Fitzpatrick17k** (Groh et al., 2021). The metadata file `fitzpatrick17k.csv` is available at
  <https://github.com/mattgroh/fitzpatrick17k>. Many of the image links in it no longer work.
  The images for this thesis were obtained from the authors of the dataset.
- **DDI** (Daneshjou et al., 2022). Available from the Stanford AIMI Center after accepting its
  research use agreement: <https://ddi-dataset.github.io>. The agreement does not allow
  redistribution, so the DDI images, the metadata and the predictions per image are not included here.

## Setup

Python 3.11 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The experiments were run with PyTorch 2.8.0 and CUDA 12.6 on an NVIDIA RTX A1000 laptop GPU.

## References

- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective number of samples. *CVPR*.
- Daneshjou, R., et al. (2022). Disparities in dermatology AI performance on a diverse, curated clinical image set. *Science Advances, 8*(32).
- Groh, M., et al. (2021). Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset. *CVPR Workshops*.
- Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. *Knowledge and Information Systems, 33*(1).
