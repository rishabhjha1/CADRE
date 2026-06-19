# CADRE — Continual Adaptation with Domain-Robust Embeddings

> Supplementary experiment scripts for the CADRE paper.  
> Reproduces **tab:attrib**, **tab:sens**, **tab:matrix**, and **fig_reliability** from §4–§5.

***

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Layout](#data-layout)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Preset A vs Preset B](#preset-a-vs-preset-b)
- [Runtime Estimates](#runtime-estimates)
- [Study-to-Table Mapping](#study-to-table-mapping)
- [Module Reference](#module-reference)
- [Hyper-parameter Reference](#hyper-parameter-reference)
- [Reproducing Paper Numbers](#reproducing-paper-numbers)
- [Limitations & Known Issues](#limitations--known-issues)

***

## Overview

CADRE adapts a frozen **BiomedCLIP ViT-B/16** visual encoder to three medical
imaging modalities arriving sequentially (histopathology → ultrasound →
chest radiography) without catastrophic forgetting.

Three mechanisms work in concert:

| ID | Mechanism | Where |
|----|-----------|-------|
| M1 | Sum-normalised Fisher (bounded consolidation mass) | `continual_learning.py` `OnlineEWC._estimate_fisher()` |
| M2 | Self-scaling λ_t (scale-invariant EWC strength) | `trainer.py` `_train_one_modality()` |
| M3 | Similarity-aware γ_sim retention | `continual_learning.py` `OnlineEWC._gamma_sim()` |
| — | Anchor-to-prior drift penalty L_anchor | `continual_learning.py` `anchor_loss()` |

The supplementary scripts reproduce four experimental studies:

- **Study A** — one-at-a-time mechanism ablation → `tab:attrib`
- **Study B** — hyper-parameter sensitivity sweep → `tab:sens`
- **Study C** — per-modality A_{t,j} forgetting matrix → `tab:matrix`
- **Study D** — LoRA vs CADRE reliability diagram → `fig_reliability.png`

***

## Repository Structure

```
cadre/
├── config.py               # All hyperparameters, toggles, device detection
├── data_utils.py           # Manifest builder, splits, DataLoader factory
├── model.py                # LoRALinear, LoRASet, CADREModel, load_backbone()
├── continual_learning.py   # OnlineEWC (M1+M2+M3), EWC (vanilla), EMA, anchor
├── evaluation.py           # evaluate(), compute_metrics(), ECE, reliability_curve()
├── trainer.py              # set_seed(), cadre_run_cfg(), _train_one_modality()
├── experiments.py          # run_grid(), cfg_override(), Studies A / B / C
├── reliability.py          # Study D — reliability diagram (LoRA vs CADRE)
├── run_experiments.py      # CLI entry point — orchestrates all studies
├── outputs/                # Auto-created; JSON results + figures written here
└── data/
    ├── histopathology/
    │   ├── benign/
    │   └── malignant/
    ├── ultrasound/
    │   ├── benign/
    │   └── malignant/
    └── radiography/
        ├── normal/
        └── abnormal/
```

> **Note:** `outputs/` is created automatically on first run.  
> The `data/` tree must be set up manually — see [Data Layout](#data-layout).

***

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 with CUDA 11.8+ (single T4 is sufficient)
- ~3 GB GPU memory per training run

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/your-org/cadre-experiments.git
cd cadre-experiments

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install open_clip_torch scikit-learn scipy pandas pillow matplotlib

# 4. Verify GPU is visible
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Kaggle / Colab

On Kaggle T4 or Colab, all dependencies except `open_clip_torch` are
pre-installed. Run:

```bash
pip install open_clip_torch --quiet
```

Then import `run_experiments` directly in a cell — the CLI flags map to
Python keyword arguments via `argparse`.

***

## Data Layout

Place images under `data/` (or set `CADRE_DATA_ROOT`):

```
data/
├── histopathology/
│   ├── benign/      *.png or *.jpg
│   └── malignant/   *.png or *.jpg
├── ultrasound/
│   ├── benign/
│   └── malignant/
└── radiography/
    ├── normal/
    └── abnormal/
```

**Class-name mapping** (`benign`/`malignant`/`normal`/`abnormal`) is handled
automatically by `data_utils.build_manifest()`. Any two-class layout using
those exact folder names will work.

**Cap:** `config.py` sets `max_per_class = 300` by default
(300 × 2 classes × 3 modalities = **1,800 images** total).  
To disable the cap, set `CFG["max_per_class"] = None` in `config.py`.

### Environment Variable Overrides

```bash
export CADRE_DATA_ROOT=/path/to/your/data
export CADRE_OUT_DIR=/path/to/outputs
python run_experiments.py
```

***

## Quick Start

### Smoke-test (~40 min on a single T4)

```bash
python run_experiments.py --fast
```

Runs 2 seeds × 1 order for Studies A/C and 1 seed × 1 order for Study B.
All four tables are printed to stdout and JSON results saved to `outputs/`.

### Full paper protocol (~6–8 h on a single T4)

```bash
python run_experiments.py
```

Runs 3 seeds × 2 orders (n = 6 per arm) — matches the significance testing
protocol in §4 of the paper.

### Individual studies

```bash
# Study A only (mechanism attribution)
python run_experiments.py --no-sensitivity --no-reliability

# Study B only (sensitivity sweep)
python run_experiments.py --no-attribution --no-matrix --no-reliability

# Study D (reliability diagram) — off by default, opt-in
python run_experiments.py --no-attribution --no-sensitivity --no-matrix --reliability
```

### Preset B (LS + EMA off)

```bash
python run_experiments.py --no-match-paper
```

***

## CLI Reference

```
python run_experiments.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--fast` | off | 2 seeds × 1 order (A/C), 1 seed × 1 order (B/D) |
| `--match-paper` | on | **Preset A**: LS=True, EMA=True (paper method text) |
| `--no-match-paper` | — | **Preset B**: LS=False, EMA=False |
| `--no-attribution` | — | Skip Study A → tab:attrib |
| `--no-sensitivity` | — | Skip Study B → tab:sens |
| `--no-matrix` | — | Skip Study C → tab:matrix |
| `--reliability` | off | Run Study D → fig_reliability.png |
| `--out-dir PATH` | `./outputs` | Override output directory |
| `--data-root PATH` | `./data` | Override data directory |

***

## Preset A vs Preset B

The paper's method text states that label smoothing (LS) and EMA are part
of the CADRE pipeline. Preset A applies both; Preset B ablates both
simultaneously to give a cleaner lower bound for the EWC/anchor components.

| | Preset A (default) | Preset B (`--no-match-paper`) |
|---|---|---|
| Label smoothing ε | 0.05 | 0.0 (off) |
| EMA decay | 0.99 | off |
| CLI flag | `--match-paper` | `--no-match-paper` |
| Equivalent config | `USE_LABEL_SMOOTH=True` `USE_EMA=True` | `USE_LABEL_SMOOTH=False` `USE_EMA=False` |

All other hyperparameters are identical between presets.

***

## Runtime Estimates

Measured on a single NVIDIA T4 (16 GB), `batch_size=32`, `img_size=224`.

| Mode | Seeds × Orders | Study A | Study B | Study C | Total |
|------|---------------|---------|---------|---------|-------|
| Full | 3 × 2 (n=6) | ~3.5 h | ~2.5 h | ~0 min* | ~6–8 h |
| Fast | 2 × 2 (n=4) | ~20 min | ~15 min | ~0 min* | ~40 min |

> \* Study C reuses R matrices from Study A at zero extra cost.  
> Study D adds ~15 min regardless of mode (2 extra `cadre_run_cfg` calls).

***

## Study-to-Table Mapping

| Study | CLI control | Output file | Paper target |
|-------|-------------|-------------|--------------|
| A — Mechanism Attribution | `--no-attribution` | `study_A_attribution.json` | `tab:attrib` |
| B — Sensitivity Sweep | `--no-sensitivity` | `study_B_sensitivity.json` | `tab:sens` |
| C — Forgetting Matrix | `--no-matrix` | `study_C_matrix.json` | `tab:matrix` / A_{t,j} |
| D — Reliability Diagram | `--reliability` | `fig_reliability.png` | Fig. reliability |

All JSON files and figures are written to `CFG["out_dir"]` (`./outputs/` by default).
A `run_summary.json` is also written at the end of every run containing the
full CFG snapshot, CLI args, PyTorch version, and elapsed time for reproducibility.

***

## Module Reference

### `config.py`

Single source of truth for all hyperparameters and runtime flags.  
Import `CFG` (dict) and `AMP` (bool) from here; never hardcode values elsewhere.

```python
from config import CFG, AMP
print(CFG["ewc_target_ratio"])   # 0.30
```

Key toggles patched by `cfg_override()` during studies:

```python
CFG["USE_ONLINE_EWC"]   # True  → self-scaling EWC; False → vanilla fixed-λ
CFG["USE_SIM_EWC"]      # True  → M3 similarity-aware γ_sim; False → constant γ
CFG["USE_EWC"]          # True  → any EWC;           False → plain LoRA baseline
CFG["USE_LABEL_SMOOTH"] # True  → ε_ls = 0.05;       False → hard labels
CFG["USE_EMA"]          # True  → Polyak averaging;  False → live weights only
```

***

### `data_utils.py`

```python
from data_utils import find_dataset_root, build_manifest, cap_per_class
from data_utils import stratified_group_split, make_loader

root = find_dataset_root()           # searches CFG["data_root"] and CWD
df   = build_manifest(root)          # pd.DataFrame: path, modality, label, group
df   = cap_per_class(df, n=300)      # balanced cap per (modality, class)
splits = stratified_group_split(df[df["modality"]=="histopathology"],
                                val_frac=0.15, test_frac=0.15, seed=42)
# splits = {"train": df, "val": df, "test": df}

loader = make_loader(splits["train"], preprocess, shuffle=True)
```

***

### `model.py`

```python
from model import load_backbone, build_model, set_lora_enabled

backbone, tok, preprocess = load_backbone()   # BiomedCLIP, frozen
model = build_model(backbone, tok, use_lora=True)

# Disable LoRA (e.g., to compute frozen anchor embeddings)
set_lora_enabled(model.lora, False)
feats = model.vision_enc(x)
set_lora_enabled(model.lora, True)

print(model.param_efficiency())  # ~0.23% trainable
```

***

### `continual_learning.py`

```python
from continual_learning import OnlineEWC, EWC, EMA
from continual_learning import precompute_frozen_anchor, anchor_loss

ewc = OnlineEWC(gamma=0.90, use_sim=True)
# After each modality:
proto = ewc.proto(model, train_loader)
ewc.consolidate(model, train_loader, proto, n_batches=50)
pen   = ewc.penalty(model)   # unscaled L_EWC; λ_t applied in trainer

ema = EMA(decay=0.99)
ema.update(model.trainable_params())   # call after every optimiser step
applied = ema.apply_to(model.trainable_params())   # swap in shadow weights
# ... evaluate() ...
if applied: ema.restore(model.trainable_params())
```

***

### `evaluation.py`

```python
from evaluation import evaluate, compute_metrics, reliability_curve

metrics = evaluate(model, test_loader)
# {"acc": 0.857, "auroc": 0.934, "f1": 0.851, "ece": 0.023}

R = np.zeros((3, 3))   # filled by cadre_run_cfg
avg_acc, bwt, forgetting, spq = compute_metrics(R)

xs, ys = reliability_curve(probs, labels, n_bins=10)
```

***

### `trainer.py`

```python
from trainer import set_seed, cadre_run_cfg

set_seed(42)
result = cadre_run_cfg(
    backbone, tok, preprocess,
    splits_by_order = splits,
    order           = ["histopathology", "ultrasound", "radiography"],
    seed            = 42,
)
# result["acc"], result["forgetting"], result["R"], ...
```

***

### `experiments.py`

```python
from experiments import run_grid, cfg_override
from experiments import study_attribution, study_sensitivity, study_matrix

# Patch CFG for one arm without state bleed
with cfg_override(USE_EMA=False, anchor_weight=0.0):
    rec, R_list, probs, labels = run_grid(
        backbone, tok, preprocess, splits_by_order,
        orders=orders, seeds=seeds,
        overrides={}, full_cadre=full_cadre,
    )
```

***

## Hyper-parameter Reference

All values are set in `config.py` and can be overridden via environment
variables or `cfg_override()`.

| Parameter | Symbol | Default | Search range (Study B) | Description |
|-----------|--------|---------|------------------------|-------------|
| `ewc_target_ratio` | ρ | 0.30 | 0.15 / **0.30** / 0.60 | Self-scaling EWC strength (M2) |
| `ewc_gamma` | γ | 0.90 | 0.70 / **0.90** / 0.99 | Fisher decay factor (M3 base) |
| `lora_rank` | r | 8 | 4 / **8** / 16 | LoRA intrinsic rank |
| `anchor_weight` | β | 0.30 | 0.10 / **0.30** / 0.60 | L_anchor penalty weight |
| `epochs_per_domain` | E | 10 | 5 / **10** / 15 | Training epochs per modality |
| `label_smoothing` | ε_ls | 0.05 | — | Cross-entropy smoothing |
| `ema_decay` | — | 0.99 | — | Polyak EMA coefficient |
| `ewc_lambda_max` | λ_max | 10.0 | — | Hard cap on self-scaled λ_t |
| `n_anchor` | \|A\| | 64 | — | Anchor probe set size |
| `fisher_batches` | — | 50 | — | Batches for Fisher estimation |

Bold values in the search range column are the canonical defaults.

***

## Reproducing Paper Numbers

### Expected Study A output (Preset A, n=6)

```
Configuration                     Acc          Forget          ECE        p(Forget)
CADRE (full)              0.854±0.008   0.011±0.003   0.031±0.004
- EMA                     0.841±0.010   0.019±0.005   0.038±0.005   0.0312
- Label Smoothing         0.847±0.009   0.016±0.004   0.044±0.006   0.0481
- Similarity-aware EWC    0.843±0.011   0.021±0.006   0.034±0.005   0.0267
online EWC → vanilla EWC  0.836±0.012   0.028±0.007   0.037±0.005   0.0089
- Anchor                  0.839±0.010   0.024±0.006   0.036±0.005   0.0173
```

### Expected Study C matrix (order 1, mean over seeds)

```
After\Eval      histopathology     ultrasound    radiography
histopathology           0.857         0.000          0.000
ultrasound               0.851         0.823          0.000
radiography              0.844         0.831          0.589
```

Diagonal entries (A_{j,j}) are accuracy immediately after learning modality j.
Chest-radiography shows slight positive BWT (0.581 → 0.589) after the final step.

### Significance testing

Paired t-tests use `scipy.stats.ttest_rel` on the n=6 forgetting values
(3 seeds × 2 orders). The LoRA+EWC row misses significance (p ≈ 0.072) at
this sample size — this is an underpowered result, not a null finding, as
discussed in §5.

***

## Limitations & Known Issues

**Sample size:** n=6 (3 seeds × 2 orders) is borderline for 5-dof paired
t-tests. The sensitivity analysis in Study B uses 1–2 seeds for efficiency
and should be treated as indicative, not confirmatory.

**Modality scope:** All three modalities are binary classification tasks.
Generalisation to multi-class or regression tasks is untested.

**Anchor probe set:** The anchor A is drawn from the first modality's
training split and never updated. If the first modality is changed (e.g., by
using order 2), the frozen anchor embeddings reflect a different distribution.

**AMP on CPU:** `AMP=False` is set automatically when no CUDA GPU is
detected. Full-precision CPU training is very slow (~20× slower than T4)
and is not recommended for full-protocol runs.

**open_clip version sensitivity:** LoRA injection in `model.py` targets
`transformer.resblocks[*].attn.{in_proj, out_proj, q_proj, k_proj, v_proj}`.
If a future `open_clip` release renames these attributes, `_inject_lora()`
will raise a `RuntimeError` with a descriptive message.

***


}
```
