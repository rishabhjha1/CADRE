"""
config.py
=========
Global configuration for CADRE experiments.
All hyperparameters, paths, and run-time toggles live here.
Override individual keys at import time or via environment variables:

    export CADRE_DATA_ROOT=/path/to/data
    export CADRE_OUT_DIR=/path/to/outputs

Or call cfg_override() (defined in experiments.py) for per-study patches.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Core dictionary
# ---------------------------------------------------------------------------
CFG = {

    # ── Paths ────────────────────────────────────────────────────────────────
    "data_root": os.environ.get("CADRE_DATA_ROOT", "./data"),
    # Expected layout:
    #   data/histopathology/{benign,malignant}/*.{png,jpg}
    #   data/ultrasound/{benign,malignant}/*.{png,jpg}
    #   data/radiography/{normal,abnormal}/*.{png,jpg}

    "out_dir": os.environ.get("CADRE_OUT_DIR", "./outputs"),
    # All result tables, matrices, and figures are written here.

    # ── Backbone ─────────────────────────────────────────────────────────────
    "backbone_name": (
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    ),
    # BiomedCLIP (Zhang et al., arXiv:2303.00915).
    # Visual encoder: ViT-B/16 frozen throughout; only LoRA + head are trained.

    # ── LoRA (PEFT) ──────────────────────────────────────────────────────────
    "lora_rank":  8,    # r — intrinsic rank of the update matrices
    "lora_alpha": 16,   # α — scales the LoRA output by α/r

    # ── Data ─────────────────────────────────────────────────────────────────
    "domain_order": ["histopathology", "ultrasound", "radiography"],
    # Canonical arrival order (order 1). Order 2 is a left-rotation:
    #   ["ultrasound", "radiography", "histopathology"]

    "max_per_class": 300,
    # Cap applied per (modality, class) cell → 300 × 2 × 3 = 1,800 images total.

    "val_frac":  0.15,  # fraction held out for validation (group-stratified)
    "test_frac": 0.15,  # fraction held out for testing   (group-stratified)
    "img_size":  224,   # spatial resolution fed to ViT-B/16

    # ── Training ─────────────────────────────────────────────────────────────
    "epochs_per_domain": 10,    # E — full passes over each modality's training set
    "lr":                5e-4,  # AdamW learning rate
    "weight_decay":      1e-4,  # AdamW weight decay (L2 regularisation)
    "batch_size":        32,    # training batch size
    "eval_batch_size":   64,    # inference batch size (no gradients; can be larger)

    "random_seeds": [42, 7, 21],
    # All seeds used in the full 3-seed × 2-order (n=6) protocol.

    # ── CADRE Regularisation ─────────────────────────────────────────────────

    # Self-scaling EWC (Online, sum-normalised Fisher):
    "ewc_target_ratio": 0.30,
    # ρ — self-scaling strength: λ_t = min(ρ · L_CE / (L_EWC + ε), λ_max)
    # Ensures the EWC penalty contributes a fixed fraction ρ of the task loss.
    # (Proposition 2, scale-invariance proof)

    "ewc_gamma": 0.90,
    # γ — base Fisher decay factor (before similarity modulation).
    # γ_sim = γ · (0.5 + 0.5 · cos(p_t, p_{t-1}))   [Eq. 3, M3]

    "ewc_lambda_max": 10.0,
    # λ_max — hard cap on the self-scaled multiplier to prevent early spikes.

    "ewc_lambda": 1.0,
    # Fixed λ used by vanilla (offline) EWC ablation only (Study A).
    # Has no effect when USE_ONLINE_EWC=True.

    "fisher_batches": 50,
    # Number of training batches used to estimate the empirical Fisher
    # after each modality.

    # Anchor-to-prior drift penalty:
    "anchor_weight": 0.30,
    # β — weight of L_anchor = ||g_LoRA(x) - g_frozen(x)||^2  [Eq. 4]

    "n_anchor": 64,
    # |A| — size of the fixed probe set drawn once from the first modality's
    # training split. Cached frozen embeddings are never re-computed.

    "use_sim_ewc": True,
    # Whether M3 (similarity-aware γ_sim modulation) is active by default.
    # Overridden per-study via the USE_SIM_EWC toggle key below.

    # ── Calibration ──────────────────────────────────────────────────────────
    "label_smoothing": 0.05,
    # ε_ls — label-smoothing coefficient applied to the cross-entropy loss.
    # Disabled (set to 0.0 internally) when USE_LABEL_SMOOTH=False.

    "ema_decay": 0.99,
    # EMA decay for Polyak weight averaging applied at evaluation only
    # (SWA-style; Izmailov et al., UAI 2018).
    # Disabled when USE_EMA=False.

    "ece_bins": 10,
    # Number of confidence bins for Expected Calibration Error (ECE).

    # ── Run-time Toggles ─────────────────────────────────────────────────────
    # These flags are patched by cfg_override() during ablation studies.
    # Preset A (matches paper method text): all True.
    # Preset B (LS + EMA off):             USE_LABEL_SMOOTH=False, USE_EMA=False.

    "USE_ONLINE_EWC":   True,
    # True  → self-scaling sum-normalised EWC (CADRE, M1+M2).
    # False → vanilla fixed-λ EWC (ablation baseline in Study A).

    "USE_SIM_EWC":      True,
    # True  → similarity-aware γ_sim retention modulation (M3).
    # False → constant γ decay (Study A M3 ablation).

    "USE_EWC":          True,
    # True  → EWC penalty active (either online or vanilla).
    # False → no EWC at all (plain LoRA / LoRA+Anchor baselines).

    "USE_LABEL_SMOOTH": True,
    # True  → apply label_smoothing ε_ls to the cross-entropy loss.
    # False → standard hard-label cross-entropy (ε_ls forced to 0).

    "USE_EMA":          True,
    # True  → Polyak EMA shadow weights applied at evaluation time.
    # False → live model weights used directly at evaluation.
}

# ---------------------------------------------------------------------------
# Derived / hardware settings
# Set once at import; never overridden by per-study cfg_override() calls.
# ---------------------------------------------------------------------------
CFG["device"] = "cuda" if torch.cuda.is_available() else "cpu"

# Mixed-precision (AMP): enabled only when a CUDA GPU is present.
# Force AMP=False to run in full fp32 (slower, useful for debugging).
AMP: bool = CFG["device"] == "cuda"

# ---------------------------------------------------------------------------
# Ensure output directory exists immediately on import
# ---------------------------------------------------------------------------
os.makedirs(CFG["out_dir"], exist_ok=True)
