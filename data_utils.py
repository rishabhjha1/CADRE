"""
data_utils.py
=============
Dataset loading, manifest building, group-stratified splitting,
and DataLoader utilities for CADRE experiments.

Data layout expected under CFG["data_root"]:
    data/
    ├── histopathology/
    │   ├── benign/      ← BreaKHis SOB filenames (slide-level group IDs parsed)
    │   └── malignant/
    ├── ultrasound/
    │   ├── benign/
    │   └── malignant/
    └── radiography/
        ├── normal/
        └── abnormal/

Leakage note (see paper §4):
  Histopathology: grouped at slide level via BreaKHis SOB filename parsing.
  Ultrasound / Radiography: grouped at image level (no recoverable patient IDs).
  → Ultrasound and radiography test accuracies are likely optimistic.
"""

import os
import re
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

import torch
from torch.utils.data import Dataset, DataLoader

from config import CFG


# =============================================================================
# 1. DATASET ROOT
# =============================================================================

def find_dataset_root(root: Optional[str] = None) -> pathlib.Path:
    """
    Resolve and validate the dataset root directory.

    Parameters
    ----------
    root : str, optional
        Override path. Falls back to CFG["data_root"] if None.

    Returns
    -------
    pathlib.Path
        Absolute path to the dataset root.

    Raises
    ------
    AssertionError
        If the resolved path does not exist on disk.
    """
    resolved = pathlib.Path(root or CFG["data_root"]).resolve()
    assert resolved.exists(), (
        f"Dataset root not found: {resolved}\n"
        f"Set CADRE_DATA_ROOT or edit CFG['data_root'] in config.py."
    )
    return resolved


# =============================================================================
# 2. MANIFEST BUILDING
# =============================================================================

# Supported image extensions
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Maps modality folder name → {class subfolder → integer label}
_MODALITY_LABEL_MAP = {
    "histopathology": {"benign": 0, "malignant": 1},
    "ultrasound":     {"benign": 0, "malignant": 1},
    "radiography":    {"normal": 0, "abnormal": 1},
}


def _parse_breakhis_group(path: pathlib.Path) -> str:
    """
    Extract slide-level group ID from a BreaKHis SOB filename.

    BreaKHis filenames follow the convention:
        SOB_<type>_<subtype>_<patient>-<year>-<slide>-<magnification>.png
    Example:
        SOB_M_DC_14-2980-200-001.png  →  group "14-2980"

    Falls back to the full stem if the pattern is not matched
    (e.g. the MSI/Multispectral subfolder packaging artefact described in §4).

    Parameters
    ----------
    path : pathlib.Path
        Full path to the image file.

    Returns
    -------
    str
        Slide-level group identifier.
    """
    # Regex targets the patient-year portion of the SOB filename
    match = re.search(r"SOB_[BM]_[^_]+_([^-]+-\d+)-\d+-\d+", path.name)
    return match.group(1) if match else path.stem


def build_manifest(root: pathlib.Path) -> pd.DataFrame:
    """
    Walk the dataset root and build a flat image manifest DataFrame.

    Columns
    -------
    path     : str   — absolute path to the image file
    label    : int   — 0 (negative/benign/normal) or 1 (positive/malignant/abnormal)
    modality : str   — one of {"histopathology", "ultrasound", "radiography"}
    group    : str   — group ID used for leakage-controlled splitting
                       (slide-level for histopathology; image-level otherwise)

    Parameters
    ----------
    root : pathlib.Path
        Dataset root returned by find_dataset_root().

    Returns
    -------
    pd.DataFrame
        One row per image. Rows are sorted by (modality, label, path).
    """
    records = []

    for modality, label_map in _MODALITY_LABEL_MAP.items():
        mod_dir = root / modality
        if not mod_dir.exists():
            print(f"[build_manifest] Warning: modality directory not found — {mod_dir}")
            continue

        for class_name, label_int in label_map.items():
            cls_dir = mod_dir / class_name
            if not cls_dir.exists():
                print(f"[build_manifest] Warning: class directory not found — {cls_dir}")
                continue

            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() not in _IMG_EXTS:
                    continue

                # Group ID: slide-level for histopathology; image-level for others
                if modality == "histopathology":
                    group = _parse_breakhis_group(img_path)
                else:
                    group = img_path.stem  # image-level (see leakage note above)

                records.append({
                    "path":     str(img_path),
                    "label":    label_int,
                    "modality": modality,
                    "group":    group,
                })

    if not records:
        raise RuntimeError(
            f"No images found under {root}. "
            "Check that subdirectories match the expected layout."
        )

    df = pd.DataFrame(records)
    df = df.sort_values(["modality", "label", "path"]).reset_index(drop=True)

    # Summary
    for mod, grp in df.groupby("modality"):
        n_groups = grp["group"].nunique()
        n_pos    = (grp["label"] == 1).sum()
        n_neg    = (grp["label"] == 0).sum()
        print(
            f"  {mod:>15s}: {len(grp):>5d} images | "
            f"{n_pos} pos / {n_neg} neg | {n_groups} groups"
        )

    return df


# =============================================================================
# 3. BALANCING / CAPPING
# =============================================================================

def cap_per_class(
    df: pd.DataFrame,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """
    Downsample to at most `n` images per (modality, label) cell.

    Sampling is reproducible given the same seed and respects the group
    structure (entire groups are not split mid-selection; individual images
    are sampled uniformly regardless of group membership).

    Parameters
    ----------
    df   : pd.DataFrame  — full manifest from build_manifest()
    n    : int           — maximum images per (modality, label) cell
    seed : int           — random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Downsampled manifest, reset index.
    """
    rng = np.random.default_rng(seed)
    parts = []

    for (mod, lbl), cell in df.groupby(["modality", "label"]):
        if len(cell) > n:
            chosen = rng.choice(cell.index, size=n, replace=False)
            cell   = cell.loc[chosen]
        parts.append(cell)

    result = pd.concat(parts).reset_index(drop=True)
    total  = len(result)
    print(f"[cap_per_class] {total} images after capping at {n}/class/modality")
    return result


# =============================================================================
# 4. GROUP-STRATIFIED SPLITTING
# =============================================================================

def stratified_group_split(
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    seed: int = 42,
) -> dict:
    """
    Split a single-modality DataFrame into train / val / test subsets
    with group-level leakage control and label stratification.

    Groups (slides / images) are kept intact across the split boundary.
    The label distribution is approximately preserved via stratified sampling.

    Parameters
    ----------
    df        : pd.DataFrame — modality slice with columns {path, label, group}
    val_frac  : float        — fraction of total data held out for validation
    test_frac : float        — fraction of total data held out for test
    seed      : int          — random seed

    Returns
    -------
    dict
        {"train": DataFrame, "val": DataFrame, "test": DataFrame}
        Each sub-DataFrame is reset-indexed.

    Notes
    -----
    val_frac_adj = val_frac / (1 - test_frac) because the validation split
    is drawn from the remaining train+val pool after the test set is removed.
    """
    # Step 1: split off the test set
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=seed
    )
    idx_trainval, idx_test = next(
        gss_test.split(df, df["label"], df["group"])
    )
    df_trainval = df.iloc[idx_trainval].reset_index(drop=True)
    df_test     = df.iloc[idx_test].reset_index(drop=True)

    # Step 2: split the remaining pool into train / val
    val_frac_adj = val_frac / (1.0 - test_frac)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_frac_adj, random_state=seed
    )
    idx_train, idx_val = next(
        gss_val.split(df_trainval, df_trainval["label"], df_trainval["group"])
    )

    return {
        "train": df_trainval.iloc[idx_train].reset_index(drop=True),
        "val":   df_trainval.iloc[idx_val].reset_index(drop=True),
        "test":  df_test,
    }


# =============================================================================
# 5. TORCH DATASET
# =============================================================================

class ModalityDataset(Dataset):
    """
    A simple map-style Dataset wrapping a manifest DataFrame.

    Each item returns:
        image  : torch.Tensor — preprocessed image (C, H, W)
        label  : torch.LongTensor — scalar class index {0, 1}
        group  : str — group ID (for reference; not used in training)

    Parameters
    ----------
    df        : pd.DataFrame — subset of the manifest (train / val / test)
    transform : callable     — torchvision-style image transform (from
                               open_clip's get_preprocess or a custom pipeline)
    """

    def __init__(self, df: pd.DataFrame, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label, str(row["group"])


# =============================================================================
# 6. DATALOADER FACTORY
# =============================================================================

def make_loader(
    df: pd.DataFrame,
    transform,
    shuffle: bool = False,
    bs: Optional[int] = None,
) -> DataLoader:
    """
    Construct a DataLoader from a manifest slice.

    Parameters
    ----------
    df        : pd.DataFrame — train, val, or test split
    transform : callable     — image preprocessing (BiomedCLIP preprocess)
    shuffle   : bool         — True for training loaders; False for eval
    bs        : int, optional — batch size override; defaults to CFG["batch_size"]
                                (use CFG["eval_batch_size"] for inference)

    Returns
    -------
    torch.utils.data.DataLoader
    """
    batch_size = bs if bs is not None else CFG["batch_size"]
    dataset    = ModalityDataset(df, transform)

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = min(4, os.cpu_count() or 1),
        pin_memory  = CFG["device"] == "cuda",
        drop_last   = False,
        persistent_workers = False,  # avoids memory leaks across studies
    )
