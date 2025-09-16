# Efficient Gait Recognition with Regularized GRU

<!-- Tech stack -->
[![TensorFlow](https://img.shields.io/badge/TensorFlow-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![snnTorch](https://img.shields.io/badge/snnTorch-2F2F2F)](https://snntorch.readthedocs.io/)

<!-- Model & Results -->
[![Best Model: GRU(H=128) ~95k params](https://img.shields.io/badge/Best%20Model-GRU(H%3D128)%20~95k%20params-brightgreen)](#)
[![Group K-Fold Accuracy: 92.42%](https://img.shields.io/badge/Group%20K--Fold-92.42%25-success)](#)
[![LOSO Accuracy: 87.33%](https://img.shields.io/badge/LOSO-87.33%25-yellowgreen)](#)
[![Seq. Length: T=100](https://img.shields.io/badge/Seq.%20Length-T%3D100-blue)](#)
[![Preprocessing: Leakage-Free](https://img.shields.io/badge/Preprocessing-Leakage%E2%80%91Free-informational)](#)


Subject-robust classification of pathological gaits from 3D skeleton kinematics (optionally fused with plantar-pressure maps), built around a compact GRU and a leakage-free preprocessing pipeline. Under subject-aware validation, the GRU achieves **92.42%** (Group K-Fold) and **87.33%** (LOSO) mean accuracy with far fewer parameters than deeper/attention models.

> **Note:** The entire pipeline—preprocessing → feature engineering → models → evaluation—is implemented in a single notebook.

---

## At a Glance

- **Signals:** per-frame **32 joints × (x, y, z) = 96 features**; optional **48×128** plantar-pressure maps.  
- **Task:** 6-class gait classification (*antalgic, lurching, normal, steppage, stiff-legged, trendelenburg*).  
- **Key idea:** regularize the **input sequence length** and enforce **leakage-free normalization**; a small GRU then rivals heavier stacks.  
- **Best model:** single-layer **GRU (H=128)** → **Dense(64, ReLU, Dropout=0.5)** → **Softmax(6)**; ~**95k** parameters.  
- **Main results:** **92.42%** (Group K-Fold), **87.33%** (LOSO); most frequent confusion is *steppage ↔ antalgic*.

---

## Pipeline (Leakage-Free)

1) **Timestamp sanitization** – ensure monotone time per trial.  
2) **Pelvis centering** (remove translation) + **stature scaling** (size normalization).  
3) **Train-only z-scoring** for positions (and velocities if used).  
4) **Optional engineered cues:**  
   - Joint **angles** (knees/hips/ankles/shoulders), scaled to \[-1, 1].  
   - **Pelvis-relative velocities**, normalized by median forward pelvis speed.  
5) **Pressure normalization** – robust percentile scaling (e.g., q0.95).  
6) **Sequence unification:** fix **T = 100** frames (keep last T), **post-pad with zeros**, and **mask** padding during training.

These choices act as strong regularizers and prevent subject leakage across folds.

---

## Feature Sets

- **Skeleton-only (96D):** standardized joint positions.  
- **Engineered fusion (~200D):** positions + angles + pelvis-relative velocities (z-scored).  
- **Multimodal (optional):** skeleton branch + CNN embedding of pressure maps with **late fusion**.

---

## Model Family Explored

Baselines (LSTM/GRU), deeper stacks, attention-augmented RNNs, CNN–LSTM hybrids, multimodal **GRU+CNN**, and a recurrent **SNN** for efficiency comparison.  
**Final classifier (skeleton-only):**  
**Masking** → **GRU(H=128)** → **Dense(64) + ReLU + Dropout(0.5)** → **Softmax(6)**.  
Rationale: gated memory for periodic-yet-variable gait, fixed-length batching without padding leakage, and moderate capacity at **T=100**.

---

## Evaluation Protocols

Strict **subject-aware** splits: Train/Val/Test by **subject**. All normalization and any unsupervised components (e.g., PCA/AE if used) are fit on **training subjects only** per fold.  
Cross-validation via **Group K-Fold (groups = subjects)** and **LOSO**.  
Metrics include overall accuracy and confusion matrices.

---

## Results (Summary)

- **Compact sequences as regularizer:** reducing variable-length inputs to **T=100** curbed overfitting and tightened train–validation gaps.  
- **Model comparison (snapshot):** the compact GRU outperformed deeper/attention and CNN–LSTM baselines while using fewer parameters; SNN was efficient but lower in accuracy.  
- **Cross-validated accuracy:** **92.42%** (Group K-Fold), **87.33%** (LOSO).

---

## Diagnostics & Insights

- **Primary confusion:** *steppage vs. antalgic*—the model often picks up upper-body compensations; temporal variability of steppage further blurs the boundary.  
- **Takeaways:**  
  1) Fixed, compact windows are powerful regularizers.  
  2) Leakage-free statistics per fold are essential.  
  3) Well-controlled inputs let simple GRUs match or beat heavier stacks.

---

## Limitations & Future Work

- Comparative (not reproduced) SOTA figures.  
- Next steps: targeted markers (e.g., foot-ground clearance), principled multimodal fusion, robustness to domain shift, and calibration/fairness analyses for clinical deployment.
