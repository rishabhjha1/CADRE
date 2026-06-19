"""
evaluation.py
=============
Evaluation utilities for CADRE experiments:

  evaluate()          — per-modality metrics (Acc, AUROC, F1, ECE)
  compute_metrics()   — continual-learning metrics from accuracy matrix R
                        (Avg. Acc, BWT, Forgetting, SPQ)
  _ece()              — Expected Calibration Error (binned)
  reliability_curve() — confidence / accuracy pairs for reliability diagrams

Metric definitions (paper §3)
------------------------------
  Avg. Acc   = (1/T) Σ_j A_{T,j}
  BWT        = (1/(T−1)) Σ_{j<T} (A_{T,j} − A_{j,j})
  Forgetting = (1/(T−1)) Σ_{j<T} (max_l A_{l,j} − A_{T,j})
  SPQ        = 2PS / (P + S)   where  P = Avg. Acc,  S = max(0, 1 − Forgetting)

  ECE        = Σ_m (|B_m|/N) · |acc(B_m) − conf(B_m)|
               (Naeini et al., AAAI 2015; Guo et al., ICML 2017)

References
----------
  McCloskey & Cohen, 1989  — catastrophic forgetting definition
  Guo et al., ICML 2017   — on calibration of modern neural networks
  Naeini et al., AAAI 2015 — obtaining well-calibrated probabilities
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader

from config import CFG, AMP


# =============================================================================
# HELPERS
# =============================================================================

def _autocast():
    """Return the appropriate autocast context for the current device."""
    if AMP:
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


# =============================================================================
# 1. PER-MODALITY EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
) -> Dict[str, float]:
    """
    Evaluate a model on a single modality's test split.

    Runs a full forward pass over the loader with no gradient computation.
    If EMA shadow weights have been applied by the trainer before calling
    this function, the evaluation reflects the EMA model.

    Parameters
    ----------
    model  : CADREModel — model in eval mode (caller must set EMA if needed)
    loader : DataLoader — test or validation loader for one modality

    Returns
    -------
    dict with keys:
        acc   : float — top-1 accuracy
        auroc : float — area under ROC curve (macro, handles binary)
        f1    : float — binary F1 score (threshold = 0.5)
        ece   : float — Expected Calibration Error (CFG["ece_bins"] bins)

    Notes
    -----
    Confidence for ECE is defined as max(p_positive, p_negative) so the
    calibration measurement is symmetric — a model that confidently predicts
    class 0 contributes to the same bin as one that confidently predicts
    class 1.
    """
    model.eval()

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for x, y, _ in loader:
        x = x.to(CFG["device"])
        with _autocast():
            logits = model(x).float()   # ensure fp32 for stable softmax
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    logits = torch.cat(all_logits, dim=0)   # (N, 2)
    labels = torch.cat(all_labels, dim=0)   # (N,)

    probs  = F.softmax(logits, dim=-1)[:, 1].numpy()   # P(class=1)
    preds  = (probs >= 0.5).astype(int)
    y_np   = labels.numpy()

    acc   = float((preds == y_np).mean())
    auroc = float(roc_auc_score(y_np, probs))
    f1    = float(f1_score(y_np, preds, zero_division=0))
    ece   = _ece(probs, y_np, n_bins=CFG["ece_bins"])

    return dict(acc=acc, auroc=auroc, f1=f1, ece=ece)


# =============================================================================
# 2. EXPECTED CALIBRATION ERROR
# =============================================================================

def _ece(
    probs:  np.ndarray,
    labels: np.ndarray,
    n_bins: int = None,
) -> float:
    """
    Compute the Expected Calibration Error (ECE).

    ECE = Σ_m (|B_m| / N) · |acc(B_m) − conf(B_m)|

    where B_m is the set of predictions whose confidence falls in the
    m-th bin of [0, 1].

    Confidence is defined as:
        conf_i = max(p_i, 1 − p_i)
    so it represents how certain the model is of its predicted class,
    regardless of which class it predicts. This is equivalent to the
    standard definition for binary classification.

    Parameters
    ----------
    probs  : np.ndarray  shape (N,) — predicted probability for class 1
    labels : np.ndarray  shape (N,) — ground-truth binary labels {0, 1}
    n_bins : int, optional — number of equal-width confidence bins
                             (defaults to CFG["ece_bins"] = 10)

    Returns
    -------
    float — ECE value ∈ [0, 1] (lower is better)
    """
    n_bins  = n_bins or CFG["ece_bins"]
    preds   = (probs >= 0.5).astype(int)
    conf    = np.where(preds == 1, probs, 1.0 - probs)   # per-sample confidence
    correct = (preds == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    N         = len(labels)
    ece       = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece     += (mask.sum() / N) * abs(bin_acc - bin_conf)

    return float(ece)


# =============================================================================
# 3. CONTINUAL-LEARNING METRICS
# =============================================================================

def compute_metrics(
    R: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute continual-learning summary metrics from the accuracy matrix R.

    R[i, j] = accuracy on modality j evaluated after training through
              modality i  (0-indexed; R is T × T).

    Metrics
    -------
    Avg. Acc   = (1/T) Σ_j R[T−1, j]
                 Average accuracy across all modalities at the end of training.

    BWT        = (1/(T−1)) Σ_{j < T−1} (R[T−1, j] − R[j, j])
                 Backward transfer: how much earlier-modality performance
                 changes after seeing subsequent modalities.
                 Positive = knowledge transfer; Negative = forgetting.

    Forgetting = (1/(T−1)) Σ_{j < T−1} (max_{l ≥ j} R[l, j] − R[T−1, j])
                 Average drop from peak performance on each earlier modality.
                 Lower is better; CADRE target: ≈ 0.011 (paper Table 1).

    SPQ        = 2 · P · S / (P + S)
                 Stability–Plasticity Quotient (harmonic mean of plasticity
                 and stability). P = Avg. Acc; S = max(0, 1 − Forgetting).
                 Higher is better; rewards models that are both accurate
                 and stable.

    Parameters
    ----------
    R : np.ndarray  shape (T, T) — accuracy matrix from a single run

    Returns
    -------
    (avg_acc, bwt, forgetting, spq) : tuple of floats

    Edge cases
    ----------
    T = 1: BWT and Forgetting are undefined → both returned as 0.0.
           SPQ = 2 · Acc · 1 / (Acc + 1) ≈ Acc for high accuracy.
    """
    T = R.shape[0]

    # ── Avg. Acc ──────────────────────────────────────────────────────────────
    avg_acc = float(R[-1].mean())

    # ── BWT and Forgetting ────────────────────────────────────────────────────
    if T <= 1:
        bwt       = 0.0
        forgetting = 0.0
    else:
        # BWT: signed change from when the modality was first learned
        bwt = float(
            np.mean([R[-1, j] - R[j, j] for j in range(T - 1)])
        )

        # Forgetting: drop from peak performance
        # peak for modality j = max accuracy observed on j across all steps
        # where j has been seen (i.e., rows i >= j)
        forgetting_per_modality = []
        for j in range(T - 1):
            # R[i, j] is valid for i >= j (modality j seen at step j)
            peak = max(R[i, j] for i in range(j, T))
            drop = peak - R[-1, j]
            forgetting_per_modality.append(drop)
        forgetting = float(np.mean(forgetting_per_modality))

    # ── SPQ ───────────────────────────────────────────────────────────────────
    plasticity = avg_acc
    stability  = max(0.0, 1.0 - forgetting)
    denom      = plasticity + stability
    spq        = float(2.0 * plasticity * stability / denom) if denom > 1e-12 else 0.0

    return avg_acc, bwt, forgetting, spq


# =============================================================================
# 4. PER-CLASS AND PER-MODALITY BREAKDOWN
# =============================================================================

@torch.no_grad()
def evaluate_per_class(
    model:  nn.Module,
    loader: DataLoader,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Per-class recall breakdown for a single modality split.

    Used to reproduce the per-modality, per-class analysis in §5 of the
    paper (e.g. CADRE chest-radiography abnormal-class recall = 0.607
    vs. 0.389–0.511 for LoRA baselines).

    Parameters
    ----------
    model       : CADREModel
    loader      : DataLoader — test split for one modality
    class_names : list[str], optional — labels for output keys
                  defaults to ["class_0", "class_1"]

    Returns
    -------
    dict  {class_name: recall_float}  plus "overall_acc"
    """
    if class_names is None:
        class_names = ["class_0", "class_1"]

    model.eval()
    all_preds:  List[int] = []
    all_labels: List[int] = []

    for x, y, _ in loader:
        x = x.to(CFG["device"])
        with _autocast():
            logits = model(x).float()
        probs  = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds  = (probs >= 0.5).astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.numpy().tolist())

    preds_np  = np.array(all_preds)
    labels_np = np.array(all_labels)

    result: Dict[str, float] = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels_np == cls_idx
        if mask.sum() == 0:
            result[cls_name] = float("nan")
        else:
            result[cls_name] = float((preds_np[mask] == cls_idx).mean())

    result["overall_acc"] = float((preds_np == labels_np).mean())
    return result


# =============================================================================
# 5. RELIABILITY CURVE  (Study D — calibration diagram)
# =============================================================================

def reliability_curve(
    probs:  np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute binned (mean_confidence, mean_accuracy) pairs for a
    reliability / calibration diagram.

    Each bin covers an equal-width interval of the confidence range [0, 1].
    Empty bins are omitted from the output so the curve has no gaps.

    Parameters
    ----------
    probs  : np.ndarray  shape (N,) — predicted P(class=1) for all test images
             (concatenated across modalities for a pooled diagram)
    labels : np.ndarray  shape (N,) — ground-truth binary labels
    n_bins : int — number of equal-width bins (default 10)

    Returns
    -------
    xs : np.ndarray — mean confidence per non-empty bin
    ys : np.ndarray — mean accuracy per non-empty bin

    Usage
    -----
    xs, ys = reliability_curve(probs, labels)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(xs, ys, 'o-', label='Model')

    A well-calibrated model has xs ≈ ys (points lie on the diagonal).
    Points above the diagonal: under-confident.
    Points below the diagonal: over-confident.
    """
    preds   = (probs >= 0.5).astype(int)
    conf    = np.where(preds == 1, probs, 1.0 - probs)
    correct = (preds == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys    = [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        xs.append(float(conf[mask].mean()))
        ys.append(float(correct[mask].mean()))

    return np.array(xs), np.array(ys)


# =============================================================================
# 6. SUMMARY PRINTER  (convenience for run_experiments.py)
# =============================================================================

def print_metrics_table(
    rows: List[Tuple[str, Dict[str, list]]],
    base_name: str = "CADRE (full)",
) -> None:
    """
    Print a formatted metrics table matching the paper's Table 1 layout.

    Parameters
    ----------
    rows      : list of (name, rec) where rec is a dict of metric lists
                as returned by run_grid() in experiments.py
    base_name : str — name of the baseline row used for paired p-value
                computation (forgetting column)
    """
    from scipy.stats import ttest_rel, sem as scipy_sem

    def _ms(vals: list) -> str:
        if len(vals) == 1:
            return f"{np.mean(vals):.3f}"
        return f"{np.mean(vals):.3f}±{scipy_sem(vals):.3f}"

    header = (
        f"{'Method':35s} {'Acc':>14s} {'AUROC':>14s} "
        f"{'BWT':>10s} {'Forget':>10s} {'SPQ':>10s}  p(Forget)"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # Find base row for p-value computation
    base_rec = next(
        (rec for name, rec in rows if name == base_name), None
    )

    for name, rec in rows:
        p_str = ""
        if (
            base_rec is not None
            and rec is not base_rec
            and len(rec["forgetting"]) == len(base_rec["forgetting"])
            and len(base_rec["forgetting"]) > 1
        ):
            _, pv = ttest_rel(base_rec["forgetting"], rec["forgetting"])
            p_str = f"{pv:.4f}"

        print(
            f"{name:35s} "
            f"{_ms(rec['acc']):>14s} "
            f"{_ms(rec['auroc']):>14s} "
            f"{_ms(rec['bwt']):>10s} "
            f"{_ms(rec['forgetting']):>10s} "
            f"{_ms(rec['spq']):>10s}  "
            f"{p_str}"
        )

    print("=" * len(header) + "\n")
