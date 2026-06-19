"""
run_experiments.py
==================
CLI entry point for all CADRE supplementary experiments.

Orchestrates Studies A–D in order, sharing the backbone, preprocessor,
and split data across all studies to avoid redundant loading.

Usage
-----
  # Full paper protocol (3 seeds × 2 orders, ~6–8 h on T4)
  python run_experiments.py

  # Fast smoke-test (2 seeds × 2 orders, ~40 min on T4)
  python run_experiments.py --fast

  # Preset B: LS + EMA off (matches MATCH_PAPER_METHOD_TEXT=False)
  python run_experiments.py --no-match-paper

  # Skip specific studies
  python run_experiments.py --no-attribution
  python run_experiments.py --no-sensitivity
  python run_experiments.py --no-matrix

  # Enable optional Study D (reliability diagram)
  python run_experiments.py --reliability

  # Combine flags
  python run_experiments.py --fast --no-sensitivity --reliability

Runtime estimates (Kaggle T4 × 1, batch_size=32)
--------------------------------------------------
  Full protocol  : ~6–8 h   (3 seeds × 2 orders × ~6 study arms)
  Fast mode      : ~40 min  (2 seeds × 1 order  × ~6 study arms)
  Study A alone  : ~3 h     (full) / ~20 min (fast)
  Study B alone  : ~2 h     (full) / ~15 min (fast)

Output files (CFG["out_dir"] = ./outputs by default)
-----------------------------------------------------
  study_A_attribution.json   — tab:attrib metric dicts
  study_B_sensitivity.json   — tab:sens sweep results
  study_C_matrix.json        — tab:matrix R mean matrix
  fig_reliability.png        — Study D calibration figure (optional)
  run_summary.json           — top-level run metadata + CLI args

Transcription guide
-------------------
  After the run, copy printed numbers directly into the LaTeX tables:
    tab:attrib  ← Study A stdout
    tab:sens    ← Study B stdout
    tab:matrix  ← Study C stdout
  For fig_reliability.png, include outputs/fig_reliability.png in the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import torch

from config import CFG, AMP
from data_utils import (
    find_dataset_root,
    build_manifest,
    cap_per_class,
    stratified_group_split,
    make_loader,
)
from model import load_backbone
from experiments import (
    cfg_override,
    run_grid,
    study_attribution,
    study_sensitivity,
    study_matrix,
)
from reliability import study_reliability


# =============================================================================
# 1. ARGUMENT PARSER
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "run_experiments.py",
        description = "CADRE supplementary experiments harness (Studies A–D).",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog      = __doc__,
    )

    # ── Preset / method text ──────────────────────────────────────────────────
    p.add_argument(
        "--match-paper", dest="match_paper",
        action="store_true", default=True,
        help=(
            "Preset A: LS=True, EMA=True — matches paper method text. "
            "[default]"
        ),
    )
    p.add_argument(
        "--no-match-paper", dest="match_paper",
        action="store_false",
        help=(
            "Preset B: LS=False, EMA=False — ablation-consistent preset "
            "that disables both LS and EMA globally."
        ),
    )

    # ── Speed ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--fast",
        action="store_true", default=False,
        help=(
            "Fast mode: 2 seeds × 1 order (Study A/C) or 1 seed × 1 order "
            "(Study B). Useful for smoke-testing on a single T4 (~40 min)."
        ),
    )

    # ── Study toggles ─────────────────────────────────────────────────────────
    p.add_argument(
        "--no-attribution", dest="run_attribution",
        action="store_false", default=True,
        help="Skip Study A (mechanism attribution → tab:attrib).",
    )
    p.add_argument(
        "--no-sensitivity", dest="run_sensitivity",
        action="store_false", default=True,
        help="Skip Study B (hyper-parameter sensitivity → tab:sens).",
    )
    p.add_argument(
        "--no-matrix", dest="run_matrix",
        action="store_false", default=True,
        help="Skip Study C (forgetting matrix → tab:matrix).",
    )
    p.add_argument(
        "--reliability", dest="run_reliability",
        action="store_true", default=False,
        help=(
            "Run Study D (reliability diagrams → fig_reliability.png). "
            "Off by default; adds ~2 extra cadre_run_cfg() calls."
        ),
    )

    # ── Output directory override ─────────────────────────────────────────────
    p.add_argument(
        "--out-dir", dest="out_dir",
        type=str, default=None,
        help=(
            "Override CFG['out_dir']. "
            "Defaults to ./outputs (or CADRE_OUT_DIR env var)."
        ),
    )

    # ── Data root override ────────────────────────────────────────────────────
    p.add_argument(
        "--data-root", dest="data_root",
        type=str, default=None,
        help=(
            "Override CFG['data_root']. "
            "Defaults to ./data (or CADRE_DATA_ROOT env var)."
        ),
    )

    return p


# =============================================================================
# 2. FULL CADRE CONFIG BUILDER
# =============================================================================

def _build_full_cadre(match_paper: bool) -> Dict:
    """
    Build the canonical CADRE-full config dict.

    Preset A (match_paper=True):  LS=True, EMA=True
    Preset B (match_paper=False): LS=False, EMA=False

    All other values come directly from CFG (set in config.py).
    """
    return dict(
        USE_ONLINE_EWC    = True,
        USE_SIM_EWC       = True,
        USE_EWC           = True,
        anchor_weight     = CFG["anchor_weight"],
        USE_LABEL_SMOOTH  = bool(match_paper),
        USE_EMA           = bool(match_paper),
        ewc_target_ratio  = CFG["ewc_target_ratio"],
        ewc_gamma         = CFG["ewc_gamma"],
        lora_rank         = CFG["lora_rank"],
        epochs_per_domain = CFG["epochs_per_domain"],
    )


# =============================================================================
# 3. SEED / ORDER SCHEDULE BUILDER
# =============================================================================

def _build_schedules(fast: bool) -> Dict:
    """
    Return the (orders, seeds) schedules for each study.

    Study A / C (attribution + matrix):
        Full mode  — all SEEDS_FULL × both ORDERS
        Fast mode  — first 2 seeds × both ORDERS

    Study B (sensitivity):
        Full mode  — first 2 seeds × canonical order only
        Fast mode  — first 1 seed  × canonical order only

    Study D (reliability):
        Always     — first 1 seed  × canonical order only
                     (probabilities are pooled across modalities)

    The canonical order is ORDERS[0] (= CFG["domain_order"]).
    The rotated order  is ORDERS[1] (= domain_order left-rotated by 1).
    """
    all_seeds = list(CFG["random_seeds"])
    dom_order = list(CFG["domain_order"])
    orders    = [
        dom_order,
        dom_order[1:] + [dom_order[0]],   # left-rotation
    ]

    return dict(
        orders_attr  = orders,
        orders_sens  = [orders[0]],
        orders_rel   = [orders[0]],
        seeds_attr   = all_seeds[:2] if fast else all_seeds,
        seeds_sens   = all_seeds[:1] if fast else all_seeds[:2],
        seeds_rel    = [all_seeds[0]],
        all_orders   = orders,
    )


# =============================================================================
# 4. DATA + BACKBONE LOADER
# =============================================================================

def _load_data_and_backbone(schedules: Dict):
    """
    Load the dataset manifest, backbone, and build splits for all orders.

    The backbone is loaded once and shared (frozen) across all study arms,
    avoiding repeated disk reads and GPU memory allocations.

    Returns
    -------
    backbone, tok, preprocess, splits_by_order
    """
    print("\n" + "─" * 60)
    print("Loading dataset manifest ...")
    base = find_dataset_root()
    df   = build_manifest(base)
    df   = cap_per_class(df, CFG["max_per_class"], CFG["random_seeds"][0])
    print(f"  {len(df)} images across {df['modality'].nunique()} modalities")

    print("Loading BiomedCLIP backbone ...")
    backbone, tok, preprocess = load_backbone()

    print("Building stratified splits ...")
    splits_by_order: Dict[tuple, Dict] = {}
    for order in schedules["all_orders"]:
        splits_by_order[tuple(order)] = {
            d: stratified_group_split(
                df[df["modality"] == d],
                val_frac  = CFG["val_frac"],
                test_frac = CFG["test_frac"],
                seed      = 42,
            )
            for d in order
        }

    n_train = sum(
        len(v["train"])
        for v in splits_by_order[tuple(schedules["all_orders"][0])].values()
    )
    print(f"  Train images (order 1): {n_train}")
    print("─" * 60 + "\n")

    return backbone, tok, preprocess, splits_by_order, df


# =============================================================================
# 5. SUMMARY WRITER
# =============================================================================

def _write_run_summary(args: argparse.Namespace, elapsed: float) -> None:
    """
    Write a JSON run summary (CLI flags, config snapshot, timing) to
    CFG["out_dir"]/run_summary.json.
    """
    summary = {
        "timestamp"      : datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": round(elapsed, 1),
        "cli_args"       : vars(args),
        "device"         : CFG["device"],
        "amp_enabled"    : AMP,
        "python_version" : sys.version,
        "torch_version"  : torch.__version__,
        "cfg_snapshot"   : {
            k: v for k, v in CFG.items()
            if not callable(v)
        },
    }
    out_path = os.path.join(CFG["out_dir"], "run_summary.json")
    try:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[summary] {out_path}")
    except Exception as e:
        print(f"\n[warn] Could not write run_summary.json: {e}")


# =============================================================================
# 6. MAIN
# =============================================================================

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # ── Apply CLI path overrides ──────────────────────────────────────────────
    if args.out_dir:
        CFG["out_dir"] = args.out_dir
        os.makedirs(CFG["out_dir"], exist_ok=True)

    if args.data_root:
        CFG["data_root"] = args.data_root

    # ── Print run banner ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CADRE — Supplementary Experiments Harness")
    print("=" * 65)
    print(f"  Device  : {CFG['device'].upper()}")
    print(f"  AMP     : {AMP}")
    print(f"  Preset  : {'A (LS+EMA on)' if args.match_paper else 'B (LS+EMA off)'}")
    print(f"  Mode    : {'FAST' if args.fast else 'FULL'}")
    print(f"  Studies : "
          f"A={'yes' if args.run_attribution else 'no'}  "
          f"B={'yes' if args.run_sensitivity else 'no'}  "
          f"C={'yes' if args.run_matrix else 'no'}  "
          f"D={'yes' if args.run_reliability else 'no'}")
    print(f"  Out dir : {CFG['out_dir']}")
    print("=" * 65 + "\n")

    # ── Guard: at least one study must be enabled ─────────────────────────────
    if not any([
        args.run_attribution,
        args.run_sensitivity,
        args.run_matrix,
        args.run_reliability,
    ]):
        print("[error] All studies disabled. Enable at least one study.")
        sys.exit(1)

    t_start = time.time()

    # ── Build configs and schedules ───────────────────────────────────────────
    full_cadre = _build_full_cadre(args.match_paper)
    schedules  = _build_schedules(args.fast)

    print("Canonical CADRE config:")
    for k, v in full_cadre.items():
        print(f"  {k:<25s} = {v}")
    print(
        f"\nSeeds (attr):  {schedules['seeds_attr']}\n"
        f"Seeds (sens):  {schedules['seeds_sens']}\n"
        f"Orders (attr): {schedules['orders_attr']}\n"
        f"Orders (sens): {schedules['orders_sens']}\n"
    )

    # ── Load data + backbone ──────────────────────────────────────────────────
    backbone, tok, preprocess, splits_by_order, df = _load_data_and_backbone(
        schedules
    )

    # ── STUDY A — Mechanism Attribution ──────────────────────────────────────
    full_R_order1: list = []
    if args.run_attribution:
        t_a = time.time()
        result_a = study_attribution(
            backbone, tok, preprocess, splits_by_order,
            orders     = schedules["orders_attr"],
            seeds      = schedules["seeds_attr"],
            full_cadre = full_cadre,
        )
        full_R_order1 = result_a["full_R_order1"]
        print(f"  [Study A] elapsed: {(time.time() - t_a) / 60:.1f} min")

    # ── STUDY B — Hyper-parameter Sensitivity ────────────────────────────────
    if args.run_sensitivity:
        t_b = time.time()
        study_sensitivity(
            backbone, tok, preprocess, splits_by_order,
            orders     = schedules["orders_sens"],
            seeds      = schedules["seeds_sens"],
            full_cadre = full_cadre,
        )
        print(f"  [Study B] elapsed: {(time.time() - t_b) / 60:.1f} min")

    # ── STUDY C — Per-Modality Forgetting Matrix ──────────────────────────────
    if args.run_matrix:
        t_c = time.time()
        study_matrix(
            full_R_order1  = full_R_order1,
            order          = schedules["all_orders"][0],
            # Fallback args: only used if Study A didn't run
            backbone       = backbone       if not args.run_attribution else None,
            tok            = tok            if not args.run_attribution else None,
            preprocess     = preprocess     if not args.run_attribution else None,
            splits_by_order= splits_by_order if not args.run_attribution else None,
            seeds          = schedules["seeds_attr"] if not args.run_attribution else None,
            full_cadre     = full_cadre     if not args.run_attribution else None,
        )
        print(f"  [Study C] elapsed: {(time.time() - t_c) / 60:.1f} min")

    # ── STUDY D — Reliability Diagrams ────────────────────────────────────────
    if args.run_reliability:
        t_d = time.time()
        study_reliability(
            backbone, tok, preprocess, splits_by_order,
            order      = schedules["all_orders"][0],
            seed       = schedules["seeds_rel"][0],
            full_cadre = full_cadre,
        )
        print(f"  [Study D] elapsed: {(time.time() - t_d) / 60:.1f} min")

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"  All studies complete.  Total elapsed: {elapsed / 60:.1f} min")
    print(f"  Results written to:  {CFG['out_dir']}/")
    print("=" * 65)
    print(
        "\nTranscription guide:\n"
        "  tab:attrib  ←  study_A_attribution.json  +  stdout (Study A)\n"
        "  tab:sens    ←  study_B_sensitivity.json  +  stdout (Study B)\n"
        "  tab:matrix  ←  study_C_matrix.json       +  stdout (Study C)\n"
        "  fig_rel     ←  fig_reliability.png               (Study D)\n"
    )

    _write_run_summary(args, elapsed)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
