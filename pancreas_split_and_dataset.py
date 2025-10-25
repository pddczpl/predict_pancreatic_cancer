
"""
pancreas_split_and_dataset.py

Output:
- splits/train.csv
- splits/val.csv

Usage (split only):
python pancreas_split_and_dataset.py `
  --ct_csv ./PANCREAS_CT_PERFILE/ct_metadata.csv `
  --label_csv ./PANCREAS_CT_PERFILE/labels_metadata.csv `
  --out_dir ./splits `
  --val_ratio 0.2 `
  --seed 42

Usage (Dataset sample check):
python pancreas_split_and_dataset.py --check `
  --csv ./splits/train.csv
"""
import argparse
import os
import random
import re
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

def _infer_patient_and_slice(path_str: str) -> Tuple[str, Optional[int]]:
    """
    Fallback helper: derive (patient_id, slice_index) from path if columns are missing.
    - patient_id := parent folder name
    - slice_index := last group of digits in filename stem, if present
    """
    p = Path(path_str)
    patient = p.parent.name
    m = re.search(r'(\d+)(?=\D*$)', p.stem)
    idx = int(m.group(1)) if m else None
    return patient, idx

def _prep_df(csv_path: str, role: str) -> pd.DataFrame:
    """
    Read a metadata CSV and standardize columns.
    role ∈ {"ct","label"}
    Returns columns: [f"{role}_path", "patient_id", "slice_index"]
    """
    df = pd.read_csv(csv_path)
    # normalize path column name
    if 'saved_path' in df.columns:
        path_col = 'saved_path'
    elif 'path' in df.columns:
        path_col = 'path'
    else:
        # Heuristic: pick first column that looks like a file path
        candidates = [c for c in df.columns if df[c].astype(str).str.contains(r'[\\/]').any()]
        if not candidates:
            raise ValueError(f"{role}: cannot find a path-like column (expected 'saved_path' or 'path').")
        path_col = candidates[0]

    out = pd.DataFrame({f"{role}_path": df[path_col].astype(str)})

    # patient_id
    if 'patient_id' in df.columns:
        out['patient_id'] = df['patient_id'].astype(str)
    else:
        out['patient_id'] = out[f"{role}_path"].map(lambda s: _infer_patient_and_slice(s)[0])

    # slice_index
    if 'slice_index' in df.columns:
        out['slice_index'] = df['slice_index'].astype(int)
    else:
        out['slice_index'] = out[f"{role}_path"].map(lambda s: _infer_patient_and_slice(s)[1]).astype('Int64')

    # Drop rows with unknown slice_index
    out = out.dropna(subset=['slice_index'])
    out['slice_index'] = out['slice_index'].astype(int)
    return out

def make_pairs(ct_csv: str, label_csv: str) -> pd.DataFrame:
    """
    Inner-join CT and LABEL slices on (patient_id, slice_index).
    Returns a dataframe with: [patient_id, slice_index, ct_path, label_path]
    """
    ct = _prep_df(ct_csv, role='ct')
    lb = _prep_df(label_csv, role='label')

    merged = pd.merge(
        ct, lb, on=['patient_id', 'slice_index'], how='inner', validate='one_to_one'
    ).sort_values(['patient_id', 'slice_index']).reset_index(drop=True)

    # Rename columns nicely
    merged = merged[['patient_id', 'slice_index', 'ct_path', 'label_path']]
    return merged

def split_by_patient(pairs_df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42):
    """
    Create patient-level train/val split to avoid leakage.
    """
    rng = random.Random(seed)
    patients = sorted(pairs_df['patient_id'].unique())
    rng.shuffle(patients)

    n_val = max(1, int(round(len(patients) * val_ratio)))
    val_patients = set(patients[:n_val])
    train_patients = set(patients[n_val:])

    train_df = pairs_df[pairs_df['patient_id'].isin(train_patients)].reset_index(drop=True)
    val_df = pairs_df[pairs_df['patient_id'].isin(val_patients)].reset_index(drop=True)

    return train_df, val_df

def save_split(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)

#########################
# PyTorch Dataset
#########################
class _LazyImports:
    done = False
    @classmethod
    def ensure(cls):
        if not cls.done:
            global torch, cv2
            import torch
            import cv2
            cls.done = True

class PancreasSliceDataset:
    """
    A minimal torch.utils.data.Dataset for (CT, LABEL) slice pairs.
    - Expects a CSV with columns: ct_path, label_path, patient_id, slice_index
    - Returns (x, y, info) where:
        x: float32 tensor [C,H,W] in [0,1]
        y: float32 tensor [1,H,W] with {0,1}
        info: (patient_id, slice_index) for reference
    """
    def __init__(self, csv_path: str, augment: bool = False):
        _LazyImports.ensure()
        self.df = pd.read_csv(csv_path)
        required = {'ct_path', 'label_path', 'patient_id', 'slice_index'}
        if not required.issubset(self.df.columns):
            raise ValueError(f"CSV must contain columns: {required}")

        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load_any(self, path: str) -> np.ndarray:
        ext = Path(path).suffix.lower()
        if ext == ".npy":
            arr = np.load(path)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Read image (keeps channels if present)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {path}")
            arr = img
        return arr

    def _to_chw(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return arr[None, ...]  # [1,H,W]
        elif arr.ndim == 3:
            # OpenCV -> BGR; for grayscale saved as 3ch, keep channels
            return np.transpose(arr, (2, 0, 1))  # [C,H,W]
        else:
            raise ValueError(f"Unexpected array shape {arr.shape}")

    def __getitem__(self, idx):
        _LazyImports.ensure()  # defensive for worker processes
        row = self.df.iloc[idx]
        x = self._load_any(row['ct_path']).astype(np.float32)
        y = self._load_any(row['label_path']).astype(np.float32)

        # Binarize label trước
        y = (y > 0).astype(np.float32)

        # Normalize CT -> [0,1] an toàn với NaN/Inf
        x_max = np.nanmax(x)
        if np.isfinite(x_max) and x_max > 1.0:
            x = x / max(x_max, 255.0)  # hoặc cứ chia 255 nếu là ảnh 8-bit

        x = self._to_chw(x)
        y = self._to_chw(y)

        # Simple optional augmentations (flip)
        if self.augment:
            if random.random() < 0.5:
                x = x[..., ::-1].copy()
                y = y[..., ::-1].copy()
            if random.random() < 0.5:
                x = x[..., :, ::-1].copy()
                y = y[..., :, ::-1].copy()

        return torch.from_numpy(x), torch.from_numpy(y), (row['patient_id'], int(row['slice_index']))

def _do_split(args):
    pairs = make_pairs(args.ct_csv, args.label_csv)
    # Summary
    by_patient = pairs.groupby('patient_id')['slice_index'].count().sort_values(ascending=False)
    print(f"[INFO] Total paired slices: {len(pairs)} | Patients: {by_patient.shape[0]}")
    print(f"[INFO] Slices/patient (top 10):\n{by_patient.head(10)}")

    train_df, val_df = split_by_patient(pairs, val_ratio=args.val_ratio, seed=args.seed)
    save_split(train_df, val_df, args.out_dir)

    print(f"[OK] Wrote: {Path(args.out_dir) / 'train.csv'}  ({len(train_df)} rows)")
    print(f"[OK] Wrote: {Path(args.out_dir) / 'val.csv'}    ({len(val_df)} rows)")

def _do_check(args):
    _LazyImports.ensure()
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.csv)
    if df.empty:
        print("[WARN] CSV is empty.")
        return

    ds = PancreasSliceDataset(args.csv, augment=False)
    x, y, info = ds[0]

    print(f"[SAMPLE] patient={info[0]} slice={info[1]}")
    print(f"CT: {tuple(x.shape)} min={float(x.min()):.3f} max={float(x.max()):.3f}")
    print(f"LB: {tuple(y.shape)} unique={torch.unique(y)}")

    # Scan toàn bộ dataset xem có NaN/Inf không
    bad = 0
    for i in range(len(ds)):
        xi, yi, _ = ds[i]
        if not torch.isfinite(xi).all() or not torch.isfinite(yi).all():
            bad += 1
    print(f"[SCAN] bad samples: {bad}/{len(ds)}")

    # Visual check (will open a window if run locally)
    # x0 = x[0].numpy()
    # y0 = y[0].numpy()
    # plt.figure()
    # plt.imshow(x0, interpolation='nearest')
    # plt.title("CT")
    # plt.figure()
    # plt.imshow(y0, interpolation='nearest')
    # plt.title("Label")
    # plt.show()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct_csv", type=str, help="Path to ct_metadata.csv")
    ap.add_argument("--label_csv", type=str, help="Path to labels_metadata.csv")
    ap.add_argument("--out_dir", type=str, default="./splits")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check", action="store_true", help="Run a quick dataset sample check")
    ap.add_argument("--csv", type=str, help="CSV to check with --check")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.check:
        if not args.csv:
            raise SystemExit("--check requires --csv pointing to a split CSV")
        _do_check(args)
    else:
        if not (args.ct_csv and args.label_csv):
            raise SystemExit("Need --ct_csv and --label_csv")
        _do_split(args)