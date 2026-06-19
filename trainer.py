"""
trainer.py
==========
Core training loop for CADRE continual adaptation experiments.

Public API
----------
  set_seed(seed)          — global reproducibility seeding
  cadre_run_cfg(...)      — full continual-learning run over all modalities;
                            returns a metrics dict consumed by run_grid()

Internal helpers
----------------
  _train_one_modality()   — single-modality training loop (all regularisers)
  _collect_probs()        — pool probabilities for Study D reliability diagrams

Training objective per step (paper §3, composite objective)
-----------------------------------------------------------
  L = L^ls_CE
    + λ_t · L_EWC(θ; F, θ*)          ← self-scaling (M2) or fixed-λ (vanilla)
    + β   · L_anchor(θ; A)

  λ_t = min( ρ · L_CE / (L_EWC + ε),  λ_max )   [Eq. 2, M2 — online EWC only]
  λ_t = CFG["ewc_lambda"]                          [vanilla EWC ablation]

  EMA shadow weights are maintained during training and applied at
  evaluation time only (SWA-style); their contribution to forgetting
  reduction is attributed separately in Study A, not credited to
  consolidation.

References
----------
  AdamW:     Loshchilov & Hutter, ICLR 2019
  Cosine LR: Loshchilov & Hutter, ICLR 2017 (SGDR)
  SWA / EMA: Izmailov et al., UAI 2018
  GradScaler: PyTorch AMP documentation
"""

from __future__ import annotations

import contextlib
import gc
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG, AMP
from continual_learning import (
    OnlineEWC,
    EWC,
    EMA,
    anchor_loss,
    precompute_frozen_anchor,
)
from data_utils import make_loader
from evaluation import evaluate, compute_metrics
from model import build_model, set_lora_enabled


# =============================================================================
# HELPERS
# =============================================================================

def set_seed(seed: int) -> None:
    """
    Set all random seeds for full reproducibility.

    Covers Python, NumPy, PyTorch CPU and CUDA. Sets
    torch.backends.cudnn.deterministic = True to eliminate
    non-deterministic CUDA operations.

    Parameters
    ----------
    seed : int — random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def _autocast():
    """Return the appropriate autocast context for the current device."""
    if AMP:
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


def _free_memory() -> None:
    """Release GPU memory and run garbage collection between runs."""
    torch.cuda.empty_cache()
    gc.collect()


def _to(tensor: torch.Tensor) -> torch.Tensor:
    """Move a tensor to CFG["device"] (non-blocking for CUDA)."""
    return tensor.to(CFG["device"], non_blocking=True)


# =============================================================================
# 1. SINGLE-MODALITY TRAINING LOOP
# =============================================================================

def _train_one_modality(
    model:     nn.Module,
    train_df,
    preprocess,
    ewc:       Optional[OnlineEWC | EWC],
    anchor_x:  Optional[torch.Tensor],
    z_frozen:  Optional[torch.Tensor],
    scaler:    torch.cuda.amp.GradScaler,
    ema:       Optional[EMA],
) -> None:
    """
    Train `model` on a single modality for CFG["epochs_per_domain"] epochs.

    Applies the full CADRE composite objective:
        L = L^ls_CE  +  λ_t · L_EWC  +  β · L_anchor

    EWC warmup (50 steps, first epoch only):
        λ_t = 0 for the first 50 optimiser steps while the Fisher is being
        estimated for modality 1. This prevents the penalty from firing on
        a zero/uninitialised Fisher and avoids early gradient spikes.
        After step 50 the self-scaling formula (Eq. 2) takes over.

    Parameters
    ----------
    model      : CADREModel — model being adapted (LoRA + head trainable)
    train_df   : pd.DataFrame — training split for the current modality
    preprocess : callable — BiomedCLIP image transform
    ewc        : OnlineEWC | EWC | None
                   OnlineEWC → self-scaling λ_t (M2)
                   EWC       → fixed λ = CFG["ewc_lambda"]
                   None      → no EWC penalty (LoRA / LoRA+Anchor baselines)
    anchor_x   : torch.Tensor | None — probe images (|A|, 3, H, W) on device
    z_frozen   : torch.Tensor | None — cached frozen embeddings (|A|, embed_dim)
    scaler     : GradScaler — AMP gradient scaler (disabled in CPU mode)
    ema        : EMA | None — weight averaging (updated every optimiser step)
    """
    loader = make_loader(train_df, preprocess, shuffle=True)

    # ── Optimiser and scheduler ───────────────────────────────────────────────
    opt = torch.optim.AdamW(
        model.trainable_params(),
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
        eps          = 1e-8,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max  = CFG["epochs_per_domain"],
        eta_min = CFG["lr"] * 0.01,   # decay to 1% of initial LR
    )

    # ── Per-step flags ────────────────────────────────────────────────────────
    ls          = CFG["label_smoothing"] if CFG["USE_LABEL_SMOOTH"] else 0.0
    online      = CFG["USE_ONLINE_EWC"]
    use_anchor  = (anchor_x is not None) and (CFG["anchor_weight"] > 0)
    global_step = 0        # counts optimiser steps across all epochs
    warmup_done = False    # flips True after the 50-step EWC warmup

    for epoch in range(CFG["epochs_per_domain"]):
        model.train()

        for x, y, _ in loader:
            x, y = _to(x), _to(y)
            opt.zero_grad()

            # ── Forward pass ─────────────────────────────────────────────────
            with _autocast():
                logits = model(x)
                ce     = F.cross_entropy(logits, y, label_smoothing=ls)
                loss   = ce

                # Anchor-to-prior penalty  β · L_anchor  [Eq. 4]
                if use_anchor:
                    loss = loss + CFG["anchor_weight"] * anchor_loss(
                        model, anchor_x, z_frozen
                    )

            # ── EWC penalty  λ_t · L_EWC ─────────────────────────────────────
            # Computed outside autocast so detached scalars are in fp32.
            if ewc is not None:
                pen = ewc.penalty(model)

                if torch.is_tensor(pen) and float(pen) > 0:
                    if online:
                        # M2: self-scaling λ_t  [Eq. 2]
                        # 50-step warmup: λ_t = 0 while Fisher is uninitialised
                        # (first modality only — ewc._F is empty until
                        # consolidate() is called at the end of modality 1)
                        if not warmup_done:
                            if global_step < 50:
                                lam = 0.0
                            else:
                                warmup_done = True
                                lam = min(
                                    CFG["ewc_target_ratio"]
                                    * float(ce.detach())
                                    / (float(pen.detach()) + 1e-12),
                                    CFG["ewc_lambda_max"],
                                )
                        else:
                            lam = min(
                                CFG["ewc_target_ratio"]
                                * float(ce.detach())
                                / (float(pen.detach()) + 1e-12),
                                CFG["ewc_lambda_max"],
                            )
                    else:
                        # Vanilla EWC: fixed global λ (ablation baseline)
                        lam = CFG["ewc_lambda"]

                    loss = loss + lam * pen

            # ── Backward + optimiser step ─────────────────────────────────────
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.trainable_params(), max_norm=1.0
            )
            scaler.step(opt)
            scaler.update()

            # GradScaler.step() may skip the optimiser update if gradients
            # are non-finite; re-enable LoRA explicitly to guard against any
            # state where the scaler zeroed the adapter outputs.
            set_lora_enabled(model.lora, True)

            # EMA shadow update (one per optimiser step)
            if ema is not None:
                ema.update(model.trainable_params())

            global_step += 1

        sched.step()   # cosine LR step at end of each epoch


# =============================================================================
# 2. PROBABILITY COLLECTOR  (Study D — reliability diagrams)
# =============================================================================

@torch.no_grad()
def _collect_probs(
    model:  nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect pooled probabilities and labels from a single loader.

    Used by Study D to build reliability diagrams comparing plain LoRA
    vs. CADRE calibration.

    Parameters
    ----------
    model  : CADREModel — model in eval mode with EMA applied if needed
    loader : DataLoader — test split for one modality

    Returns
    -------
    probs  : np.ndarray  shape (N,) — P(class=1) for each image
    labels : np.ndarray  shape (N,) — ground-truth binary labels
    """
    model.eval()
    ps, ys = [], []

    for x, y, _ in loader:
        with _autocast():
            logits = model(_to(x)).float()
        ps.append(F.softmax(logits, dim=-1)[:, 1].cpu())
        ys.append(y)

    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


# =============================================================================
# 3. FULL CONTINUAL-LEARNING RUN
# =============================================================================

def cadre_run_cfg(
    backbone,
    tok,
    preprocess,
    splits:             Dict[str, Dict[str, object]],
    order:              List[str],
    seed:               int,
    collect_probs_flag: bool = False,
) -> Dict:
    """
    Run a complete continual-learning experiment over all modalities.

    Modalities arrive sequentially in `order`. After each modality:
      1. Fisher is consolidated (OnlineEWC) or accumulated (EWC).
      2. EMA shadow weights are applied for evaluation, then restored.
      3. All seen modalities are re-evaluated → R[i, j] populated.

    All CFG toggles (USE_EWC, USE_ONLINE_EWC, USE_SIM_EWC, USE_LABEL_SMOOTH,
    USE_EMA, anchor_weight, etc.) are read from CFG at call time, so
    cfg_override() in experiments.py controls the full study configuration.

    Parameters
    ----------
    backbone   : nn.Module  — frozen BiomedCLIP visual encoder (shared)
    tok        : any        — tokenizer (API parity; not used in visual path)
    preprocess : callable   — BiomedCLIP image transform
    splits     : dict       — {modality: {"train": df, "val": df, "test": df}}
                              keyed by modality name
    order      : list[str]  — modality arrival order for this run
    seed       : int        — random seed (controls weight init + data shuffle)
    collect_probs_flag : bool
                            — if True, pools P(class=1) and labels from the
                              final evaluation step for Study D reliability
                              diagrams. Only meaningful for order[0], seed[0].

    Returns
    -------
    dict with keys:
        acc        : float        — Avg. Acc across all modalities (final step)
        bwt        : float        — Backward Transfer
        forgetting : float        — Average Forgetting
        spq        : float        — Stability–Plasticity Quotient
        auroc      : float        — mean AUROC across modalities (final step)
        f1         : float        — mean F1 across modalities (final step)
        ece        : float        — mean ECE across modalities (final step)
        params_pct : float        — % trainable parameters
        R          : np.ndarray   — T × T accuracy matrix
        probs      : np.ndarray   — (optional) pooled P(class=1) for Study D
        labels     : np.ndarray   — (optional) pooled ground-truth for Study D
    """
    set_seed(seed)

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(backbone, tok, use_lora=True)

    # ── Instantiate regularisers ──────────────────────────────────────────────
    # EWC
    if CFG.get("USE_EWC", True):
        if CFG["USE_ONLINE_EWC"]:
            ewc = OnlineEWC(
                gamma   = CFG["ewc_gamma"],
                use_sim = CFG["USE_SIM_EWC"],
            )
        else:
            ewc = EWC()   # vanilla fixed-λ
    else:
        ewc = None        # LoRA / LoRA+Anchor baseline

    # EMA
    ema = EMA(decay=CFG["ema_decay"]) if CFG["USE_EMA"] else None

    # AMP scaler (no-op when AMP=False)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    # ── Anchor probe set ──────────────────────────────────────────────────────
    # Fixed from the first modality's training split. Cached frozen embeddings
    # are computed once and reused across all modalities (Eq. 4).
    anchor_x = z_frozen = None
    if CFG["anchor_weight"] > 0:
        first_loader = make_loader(
            splits[order[0]]["train"], preprocess, shuffle=False
        )
        for x, _, _ in first_loader:
            anchor_x = _to(x[: CFG["n_anchor"]])
            break
        z_frozen = precompute_frozen_anchor(model, anchor_x)

    # ── Accuracy matrix R ─────────────────────────────────────────────────────
    T          = len(order)
    R          = np.zeros((T, T))
    final_auroc, final_f1, final_ece = [], [], []
    pooled_p,  pooled_y              = [], []

    # ── Sequential modality loop ──────────────────────────────────────────────
    for i, domain in enumerate(order):

        # ── Train on current modality ─────────────────────────────────────────
        _train_one_modality(
            model      = model,
            train_df   = splits[domain]["train"],
            preprocess = preprocess,
            ewc        = ewc,
            anchor_x   = anchor_x,
            z_frozen   = z_frozen,
            scaler     = scaler,
            ema        = ema,
        )

        # ── Fisher consolidation ──────────────────────────────────────────────
        if ewc is not None:
            train_loader = make_loader(
                splits[domain]["train"], preprocess, shuffle=False
            )
            if CFG["USE_ONLINE_EWC"]:
                # Compute modality prototype for M3 γ_sim
                proto = ewc.proto(model, train_loader)
                ewc.consolidate(
                    model,
                    train_loader,
                    proto,
                    n_batches = CFG["fisher_batches"],
                )
            else:
                # Vanilla EWC: no prototype, no γ_sim
                ewc.consolidate(model, train_loader)

        # ── Evaluation (all seen modalities) ──────────────────────────────────
        # Apply EMA shadow weights temporarily; restore live weights after.
        applied = (
            ema.apply_to(model.trainable_params())
            if ema is not None
            else False
        )

        for j, eval_domain in enumerate(order):
            test_loader = make_loader(
                splits[eval_domain]["test"],
                preprocess,
                shuffle   = False,
                bs        = CFG["eval_batch_size"],
            )
            metrics    = evaluate(model, test_loader)
            R[i, j]    = metrics["acc"]

            # Collect final-step metrics (after last modality i = T−1)
            if i == T - 1:
                final_auroc.append(metrics["auroc"])
                final_f1.append(metrics["f1"])
                final_ece.append(metrics["ece"])

                # Optionally pool probabilities for Study D
                if collect_probs_flag:
                    p, yv = _collect_probs(model, test_loader)
                    pooled_p.append(p)
                    pooled_y.append(yv)

        # Restore live weights after evaluation
        if applied:
            ema.restore(model.trainable_params())

        # ── Progress log ──────────────────────────────────────────────────────
        row_accs = " | ".join(
            f"{order[j][:4]}: {R[i, j]:.3f}" for j in range(i + 1)
        )
        print(f"  [step {i+1}/{T}] {domain:>15s} done → {row_accs}")

    # ── Compute continual-learning summary metrics ────────────────────────────
    avg_acc, bwt, forgetting, spq = compute_metrics(R)

    result = dict(
        acc        = avg_acc,
        bwt        = bwt,
        forgetting = forgetting,
        spq        = spq,
        auroc      = float(np.mean(final_auroc)),
        f1         = float(np.mean(final_f1)),
        ece        = float(np.mean(final_ece)),
        params_pct = model.param_efficiency(),
        R          = R,
    )

    # Attach pooled probabilities if collected (Study D)
    if collect_probs_flag and pooled_p:
        result["probs"]  = np.concatenate(pooled_p)
        result["labels"] = np.concatenate(pooled_y)

    # ── Clean up GPU memory between runs ─────────────────────────────────────
    del model
    _free_memory()

    return result
