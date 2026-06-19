"""
experiments.py
==============
Study A — Mechanism Attribution    →  tab:attrib
Study B — Hyper-parameter Sensitivity →  tab:sens
Study C — Per-Modality Forgetting Matrix →  tab:matrix

Each study is self-contained, prints its table to stdout, and returns
structured results that run_experiments.py passes to downstream consumers
(Study C reuses Study A's R matrices; Study D in reliability.py reuses
the run_grid() infrastructure).

Grid runner
-----------
  run_grid()    — executes cadre_run_cfg() over all (order × seed) combos
                  with a given config override applied on top of full_cadre.
  cfg_override() — context manager that temporarily patches CFG keys and
                   restores them on exit, preventing state bleed between arms.

Statistical testing
-------------------
  Paired t-tests (n = 6, dof = 5) on forgetting values, matching the paper's
  significance testing protocol (§4). The contrast with LoRA+EWC missing
  significance (p = 0.072) is reproduced faithfully — underpowered at n = 6,
  not null (§5).

References
----------
  Paired t-test:   scipy.stats.ttest_rel
  SEM:             scipy.stats.sem
  Paper §4–§5:     Experimental setup, Results, Component attribution
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import sem as scipy_sem, ttest_rel

from config import CFG
from trainer import cadre_run_cfg


# =============================================================================
# 1. CFG CONTEXT MANAGER
# =============================================================================

@contextlib.contextmanager
def cfg_override(**kw):
    """
    Temporarily patch CFG keys for a study arm, then restore originals.

    Prevents state bleed between back-to-back study arms in run_grid().
    Handles keys that did not previously exist in CFG (stores None as
    sentinel and deletes them on exit rather than setting to None).

    Usage
    -----
    with cfg_override(USE_EMA=False, anchor_weight=0.0):
        result = cadre_run_cfg(...)
    # CFG["USE_EMA"] and CFG["anchor_weight"] restored here

    Parameters
    ----------
    **kw : dict — CFG keys and their temporary values
    """
    # Store originals; use sentinel _MISSING for keys not in CFG
    _MISSING = object()
    old = {k: CFG.get(k, _MISSING) for k in kw}
    CFG.update(kw)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is _MISSING:
                CFG.pop(k, None)   # key didn't exist before; remove it
            else:
                CFG[k] = v


# =============================================================================
# 2. FORMATTING HELPERS
# =============================================================================

def _ms(values: list) -> str:
    """
    Format a list of metric values as 'mean±SEM' (n > 1) or 'mean' (n = 1).

    Matches the paper's Table 1 reporting convention.
    """
    if len(values) == 1:
        return f"{np.mean(values):.3f}"
    return f"{np.mean(values):.3f}±{scipy_sem(values):.3f}"


def _pval(base: list, other: list) -> str:
    """
    Compute a paired t-test p-value between two metric lists.

    Returns an empty string if lists are the same object, have
    mismatched lengths, or have fewer than 2 observations (underpowered).
    """
    if (
        base is other
        or len(base) != len(other)
        or len(base) < 2
    ):
        return ""
    _, pv = ttest_rel(base, other)
    return f"{pv:.4f}"


# =============================================================================
# 3. GRID RUNNER
# =============================================================================

def run_grid(
    backbone,
    tok,
    preprocess,
    splits_by_order:    Dict[tuple, Dict],
    orders:             List[List[str]],
    seeds:              List[int],
    overrides:          Dict,
    full_cadre:         Dict,
    collect_probs:      bool = False,
    keep_R_order1:      bool = False,
) -> Tuple[Dict[str, list], List[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run cadre_run_cfg() over all (order × seed) combinations with
    full_cadre config + overrides applied via cfg_override().

    The combined config dict is:
        effective_cfg = {**full_cadre, **overrides}

    This means overrides take precedence over full_cadre defaults,
    and both together override whatever was in CFG before the call.

    Parameters
    ----------
    backbone         : nn.Module — frozen visual encoder (shared across runs)
    tok              : any       — tokenizer
    preprocess       : callable  — BiomedCLIP image transform
    splits_by_order  : dict      — {tuple(order): {domain: {train/val/test}}}
    orders           : list      — list of modality orderings to run
    seeds            : list      — list of random seeds to run
    overrides        : dict      — CFG patches for this study arm
    full_cadre       : dict      — canonical CADRE-full config
    collect_probs    : bool      — collect P(class=1) for Study D
                                   (first order, first seed only)
    keep_R_order1    : bool      — store R matrices for orders[0]
                                   (used by Study C)

    Returns
    -------
    rec          : dict of metric lists — keys: acc, forgetting, ece,
                   auroc, f1, bwt, spq; each a list of n_runs floats
    R_order1     : list of np.ndarray  — R matrices for orders[0]
                   (empty if keep_R_order1=False)
    probs        : np.ndarray | None   — pooled P(class=1) for Study D
    labels       : np.ndarray | None   — pooled ground-truth for Study D
    """
    # Merge full_cadre baseline with this arm's overrides
    effective = {**full_cadre, **overrides}

    rec: Dict[str, list] = {
        k: [] for k in ["acc", "forgetting", "ece", "auroc", "f1", "bwt", "spq"]
    }
    R_order1:       List[np.ndarray]      = []
    probs:          Optional[np.ndarray]  = None
    labels:         Optional[np.ndarray]  = None

    with cfg_override(**effective):
        for order in orders:
            for seed in seeds:
                # Collect probs only for the very first (order, seed) combo
                do_probs = (
                    collect_probs
                    and order == orders[0]
                    and seed  == seeds[0]
                )

                result = cadre_run_cfg(
                    backbone,
                    tok,
                    preprocess,
                    splits_by_order[tuple(order)],
                    order,
                    seed,
                    collect_probs_flag=do_probs,
                )

                for k in rec:
                    rec[k].append(result[k])

                if keep_R_order1 and order == orders[0]:
                    R_order1.append(result["R"])

                if do_probs and "probs" in result:
                    probs  = result["probs"]
                    labels = result["labels"]

    return rec, R_order1, probs, labels


# =============================================================================
# 4. STUDY A — MECHANISM ATTRIBUTION
# =============================================================================

def study_attribution(
    backbone,
    tok,
    preprocess,
    splits_by_order: Dict[tuple, Dict],
    orders:          List[List[str]],
    seeds:           List[int],
    full_cadre:      Dict,
) -> Dict:
    """
    Study A: ablate each CADRE component one at a time.

    Rows produced (in order):
      CADRE (full)              — canonical config, all components ON
      - EMA                     — USE_EMA=False            (if EMA was ON)
      - Label Smoothing         — USE_LABEL_SMOOTH=False   (if LS was ON)
      - Similarity-aware EWC    — USE_SIM_EWC=False        (if SIM was ON)
      online EWC → vanilla EWC  — USE_ONLINE_EWC=False, USE_SIM_EWC=False
      - Anchor                  — anchor_weight=0.0

    Each ablation row runs run_grid() with the single override applied on
    top of full_cadre; CADRE-full is also the reference for paired t-tests.

    The CADRE-full run also harvests order-1 R matrices for Study C via
    keep_R_order1=True, avoiding a redundant re-run.

    Parameters
    ----------
    backbone / tok / preprocess / splits_by_order : see run_grid()
    orders      : list of orderings (canonical + rotated)
    seeds       : list of seeds for attribution (ATTR_SEEDS)
    full_cadre  : canonical CADRE config dict

    Returns
    -------
    dict with keys:
        rows         : list of (name, rec) tuples — each rec is a metric dict
        full_R_order1: list of R matrices for CADRE-full on orders[0]
                       (reused by study_matrix() without re-running)
    """
    print("\n" + "=" * 65)
    print("STUDY A: Mechanism Attribution  →  tab:attrib")
    print("=" * 65)

    rows:          List[Tuple[str, Dict]] = []
    full_R_order1: List[np.ndarray]       = []

    # ── CADRE (full) ──────────────────────────────────────────────────────────
    print("\n  [A] CADRE (full)")
    full_rec, full_R_order1, _, _ = run_grid(
        backbone, tok, preprocess, splits_by_order,
        orders, seeds,
        overrides     = {},
        full_cadre    = full_cadre,
        keep_R_order1 = True,
    )
    rows.append(("CADRE (full)", full_rec))

    # ── Component ablations ───────────────────────────────────────────────────
    # Only ablate components that are actually ON in full_cadre to avoid
    # redundant "no-op" ablation rows (e.g. if Preset B has EMA=False,
    # the "-EMA" row would be identical to full and misleading).

    candidates: List[Tuple[str, Dict]] = []

    if full_cadre.get("USE_EMA", True):
        candidates.append((
            "- EMA",
            {"USE_EMA": False},
        ))

    if full_cadre.get("USE_LABEL_SMOOTH", True):
        candidates.append((
            "- Label Smoothing",
            {"USE_LABEL_SMOOTH": False},
        ))

    if full_cadre.get("USE_SIM_EWC", True):
        candidates.append((
            "- Similarity-aware EWC",
            {"USE_SIM_EWC": False},
        ))

    # EWC redesign: revert self-scaling → vanilla fixed-λ
    # Disables both M2 (online) and M3 (sim) to isolate the redesign effect
    candidates.append((
        "online EWC → vanilla EWC",
        {"USE_ONLINE_EWC": False, "USE_SIM_EWC": False},
    ))

    # Anchor ablation (CADRE - anchor control in paper Table 1)
    candidates.append((
        "- Anchor",
        {"anchor_weight": 0.0},
    ))

    for name, override in candidates:
        print(f"  [A] {name}")
        rec, _, _, _ = run_grid(
            backbone, tok, preprocess, splits_by_order,
            orders, seeds,
            overrides  = override,
            full_cadre = full_cadre,
        )
        rows.append((name, rec))

    # ── Print tab:attrib ──────────────────────────────────────────────────────
    _print_attribution_table(rows)

    # ── Save to JSON ──────────────────────────────────────────────────────────
    _save_study_results(
        "study_A_attribution.json",
        {name: {k: [float(v) for v in vals] for k, vals in rec.items()}
         for name, rec in rows},
    )

    return {"rows": rows, "full_R_order1": full_R_order1}


def _print_attribution_table(rows: List[Tuple[str, Dict]]) -> None:
    """Print Study A results in paper Table 1 format with p-values."""
    base_rec = rows[0][1]   # CADRE (full) is always first

    sep  = "=" * 80
    hdr  = (
        f"{'Configuration':35s} "
        f"{'Acc':>14s} "
        f"{'Forget':>12s} "
        f"{'ECE':>10s} "
        f"{'BWT':>10s} "
        f"  p(Forget)"
    )
    print(f"\n{sep}")
    print("tab:attrib  —  mean ± SEM  (paired p vs CADRE-full on forgetting)")
    print(sep)
    print(hdr)
    print("-" * 80)

    for name, rec in rows:
        pv = _pval(base_rec["forgetting"], rec["forgetting"])
        print(
            f"{name:35s} "
            f"{_ms(rec['acc']):>14s} "
            f"{_ms(rec['forgetting']):>12s} "
            f"{_ms(rec['ece']):>10s} "
            f"{_ms(rec['bwt']):>10s} "
            f"  {pv}"
        )

    print(sep + "\n")


# =============================================================================
# 5. STUDY B — HYPER-PARAMETER SENSITIVITY
# =============================================================================

def study_sensitivity(
    backbone,
    tok,
    preprocess,
    splits_by_order: Dict[tuple, Dict],
    orders:          List[List[str]],
    seeds:           List[int],
    full_cadre:      Dict,
) -> None:
    """
    Study B: one-at-a-time sweep over five hyper-parameters.

    Sweeps (label, CFG key, [low, mid, high]):
      Self-scale ratio ρ    ewc_target_ratio   [0.15, 0.30, 0.60]
      Retention γ           ewc_gamma          [0.70, 0.90, 0.99]
      LoRA rank r           lora_rank          [4,    8,    16  ]
      Anchor weight β       anchor_weight      [0.10, 0.30, 0.60]
      Epochs per modality E epochs_per_domain  [5,    10,   15  ]

    The mid-point value coincides with the canonical CADRE config
    (full_cadre defaults) and is computed once via run_grid({}) then
    reused for all five sweep tables — avoiding 5 redundant re-runs.

    Expected behaviour (from paper §5 / hyper-parameter sensitivity):
      - Accuracy and forgetting stable across the mid-range of all factors.
      - Notable degradations only at extremes:
          E=15   → forgetting 0.065 (over-fitting)
          γ=0.99 → forgetting 0.031 (over-retention)
          E=5    → accuracy  0.702  (under-fitting)

    Parameters
    ----------
    (same as study_attribution)
    seeds / orders : typically SENS_SEEDS (1–2) × SENS_ORDERS (1 order)
                     for the indicative single-seed/order sweep described
                     in the paper.
    """
    print("\n" + "=" * 65)
    print("STUDY B: Hyper-parameter Sensitivity  →  tab:sens")
    print("=" * 65)

    sweeps: List[Tuple[str, str, List]] = [
        ("Self-scale ratio rho",  "ewc_target_ratio",  [0.15, 0.30, 0.60]),
        ("Retention gamma",       "ewc_gamma",         [0.70, 0.90, 0.99]),
        ("LoRA rank r",           "lora_rank",         [4,    8,    16  ]),
        ("Anchor weight beta",    "anchor_weight",     [0.10, 0.30, 0.60]),
        ("Epochs per modality E", "epochs_per_domain", [5,    10,   15  ]),
    ]

    # Compute mid-point baseline once (shared across all sweep rows)
    print("  [B] Computing mid-point baseline ...")
    mid_rec, _, _, _ = run_grid(
        backbone, tok, preprocess, splits_by_order,
        orders, seeds,
        overrides  = {},
        full_cadre = full_cadre,
    )

    # Collect all results for JSON export
    sensitivity_results: Dict[str, Dict] = {}

    sep = "=" * 70
    hdr = (
        f"{'Factor':28s} "
        f"{'Setting':>10s} "
        f"{'Acc':>14s} "
        f"{'Forget':>12s} "
        f"{'ECE':>10s}"
    )
    print(f"\n{sep}")
    print("tab:sens  —  one-at-a-time sweep  (mean ± SEM)")
    print(sep)
    print(hdr)
    print("-" * 70)

    for label, key, vals in sweeps:
        mid_val = vals[1]   # canonical default is always the middle value
        sensitivity_results[label] = {}

        for v in vals:
            if v == mid_val:
                rec = mid_rec   # reuse pre-computed mid-point
            else:
                rec, _, _, _ = run_grid(
                    backbone, tok, preprocess, splits_by_order,
                    orders, seeds,
                    overrides  = {key: v},
                    full_cadre = full_cadre,
                )

            marker = " *" if v == mid_val else "  "   # mark canonical default
            print(
                f"{label:28s} "
                f"{str(v):>10s}{marker}"
                f"{_ms(rec['acc']):>14s} "
                f"{_ms(rec['forgetting']):>12s} "
                f"{_ms(rec['ece']):>10s}"
            )

            sensitivity_results[label][str(v)] = {
                k: [float(x) for x in vals_]
                for k, vals_ in rec.items()
            }

    print(sep)
    print("  * = canonical CADRE default\n")

    _save_study_results("study_B_sensitivity.json", sensitivity_results)


# =============================================================================
# 6. STUDY C — PER-MODALITY FORGETTING MATRIX
# =============================================================================

def study_matrix(
    full_R_order1: List[np.ndarray],
    order:         List[str],
    backbone       = None,
    tok            = None,
    preprocess     = None,
    splits_by_order: Dict[tuple, Dict] = None,
    seeds:         List[int] = None,
    full_cadre:    Dict = None,
) -> np.ndarray:
    """
    Study C: print the mean A_{t,j} accuracy matrix for CADRE-full on order 1.

    If full_R_order1 is non-empty (Study A already ran), the matrices are
    averaged directly. If it is empty (Study A was skipped via --no-attribution),
    a fresh order-1 run is performed using the provided backbone/splits/seeds.

    The matrix rows correspond to "after training through modality i" and
    columns to "evaluated on modality j". The diagonal R[j,j] is the
    accuracy immediately after learning modality j; off-diagonal entries
    in the final row show retention at the end of training.

    Expected pattern for CADRE (paper §5, Fig. At,j):
      Histopathology remains near-flat across updates (0.857 → 0.870 → 0.844).
      Chest-radiography improves after the later modality (0.581 → 0.589)
      — positive backward transfer.

    Parameters
    ----------
    full_R_order1    : list of R matrices from Study A (may be empty)
    order            : canonical modality order (order 1)
    backbone / tok / preprocess / splits_by_order / seeds / full_cadre :
                       required only if full_R_order1 is empty

    Returns
    -------
    np.ndarray  shape (T, T) — mean accuracy matrix
    """
    print("\n" + "=" * 65)
    print("STUDY C: A_{t,j} Matrix  (CADRE full, order 1)  →  tab:matrix")
    print("=" * 65)

    # ── Compute or reuse R matrices ───────────────────────────────────────────
    if not full_R_order1:
        if any(v is None for v in [backbone, tok, preprocess,
                                    splits_by_order, seeds, full_cadre]):
            raise ValueError(
                "study_matrix(): full_R_order1 is empty and no backbone/"
                "splits/seeds/full_cadre provided. Either run Study A first "
                "or pass all required arguments."
            )
        print("  [C] Study A not run — computing order-1 R matrices now ...")
        _, full_R_order1, _, _ = run_grid(
            backbone, tok, preprocess, splits_by_order,
            [order], seeds,
            overrides     = {},
            full_cadre    = full_cadre,
            keep_R_order1 = True,
        )

    # ── Average across seeds ──────────────────────────────────────────────────
    Rmat = np.mean(np.stack(full_R_order1, axis=0), axis=0)   # (T, T)

    # ── Print table ───────────────────────────────────────────────────────────
    col_w = 16
    header = "After \\ Eval   " + "".join(f"{d:>{col_w}s}" for d in order)
    sep    = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for i, d in enumerate(order):
        row = "".join(f"{Rmat[i, j]:>{col_w}.3f}" for j in range(len(order)))
        # Mark the diagonal (A_{j,j}) with an asterisk
        row_marked = ""
        for j in range(len(order)):
            val = f"{Rmat[i, j]:.3f}"
            if i == j:
                val = val + "*"
            row_marked += f"{val:>{col_w}s}"
        print(f"{d:<14s}{row_marked}")
    print(sep)
    print("  * = A_{j,j}  (accuracy right after learning modality j)\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_study_results(
        "study_C_matrix.json",
        {"order": order, "R_mean": Rmat.tolist()},
    )

    return Rmat


# =============================================================================
# 7. IO HELPER
# =============================================================================

def _save_study_results(filename: str, data: dict) -> None:
    """
    Serialise study results to JSON in CFG["out_dir"].

    Silently skips on serialisation errors to avoid crashing a long
    multi-study run over a minor IO issue.
    """
    out_path = os.path.join(CFG["out_dir"], filename)
    try:
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [saved] {out_path}")
    except Exception as e:
        print(f"  [warn] Could not save {out_path}: {e}")
