#!/usr/bin/env python3
"""
train_classifier.py

Train a binary highlight classifier from per-segment embeddings saved as .npz.

Expected each .npz to contain:
  - y: int array shape [N] (0/1)
  - start_sec, end_sec: float arrays shape [N] (optional)
  - X_video: float array shape [N, Dv] (optional if audio-only)
  - X_audio: float array shape [N, Da] (optional if video-only)

Example:
  python scripts/train_classifier.py \
    --features_dir data/features \
    --holdout_vod vod_003 \
    --model mlp \
    --out_path data/output/model.pt

Then later:
  python scripts/predict.py --model_path data/output/model.pt ...
"""

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

# Optional sklearn baseline
try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Utils
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", type=str, required=True, help="Directory containing *.npz")
    p.add_argument("--holdout_vod", type=str, required=True, help="e.g. vod_003 (file is vod_003.npz)")
    p.add_argument("--model", type=str, default="mlp", choices=["mlp", "logreg"])
    p.add_argument("--out_path", type=str, required=True)

    # Feature selection
    p.add_argument("--use_audio", action="store_true", help="Use audio features (default: use if present)")
    p.add_argument("--use_video", action="store_true", help="Use video features (default: use if present)")
    p.add_argument("--audio_only", action="store_true")
    p.add_argument("--video_only", action="store_true")

    # Training
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for metrics")

    # Runtime
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_metrics_json", type=str, default="", help="Optional path to save metrics JSON")

    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class VodData:
    vod_id: str
    X: np.ndarray   # [N, D]
    y: np.ndarray   # [N]


def load_npz(path: str, vod_id: str, want_audio: bool, want_video: bool) -> VodData:
    d = np.load(path)
    if "y" not in d.files:
        raise ValueError(f"{path} missing key 'y'")

    y = d["y"].astype(np.int64)
    parts: List[np.ndarray] = []

    if want_video:
        if "X_video" not in d.files:
            raise ValueError(f"{path}: want_video=True but no X_video found.")
        parts.append(d["X_video"].astype(np.float32))

    if want_audio:
        if "X_audio" not in d.files:
            raise ValueError(f"{path}: want_audio=True but no X_audio found.")
        parts.append(d["X_audio"].astype(np.float32))

    if not parts:
        raise ValueError("No features selected. Use --audio_only / --video_only / --use_audio / --use_video.")

    # Concatenate along feature dim
    X = np.concatenate(parts, axis=1).astype(np.float32)

    # Basic sanity
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{path}: X rows {X.shape[0]} != y rows {y.shape[0]}")

    # Replace NaNs/Infs defensively
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return VodData(vod_id=vod_id, X=X, y=y)


def choose_feature_flags(args: argparse.Namespace, sample_npz: str) -> Tuple[bool, bool]:
    """
    Decide whether to use audio/video based on flags and availability.
    """
    if args.audio_only and args.video_only:
        raise ValueError("Choose only one of --audio_only/--video_only")

    if args.audio_only:
        return True, False
    if args.video_only:
        return False, True

    # If user explicitly requested use_audio/use_video, obey
    if args.use_audio or args.use_video:
        return bool(args.use_audio), bool(args.use_video)

    # Otherwise auto-detect based on keys in one file
    d = np.load(sample_npz)
    has_a = "X_audio" in d.files
    has_v = "X_video" in d.files
    if not (has_a or has_v):
        raise ValueError(f"{sample_npz} has neither X_audio nor X_video.")
    return has_a, has_v


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score using train mean/std only.
    Returns standardized train/test and (mean, std).
    """
    mu = X_train.mean(axis=0, keepdims=True)
    sig = X_train.std(axis=0, keepdims=True)
    sig = np.where(sig < 1e-8, 1.0, sig)
    Xtr = (X_train - mu) / sig
    Xte = (X_test - mu) / sig
    return Xtr.astype(np.float32), Xte.astype(np.float32), mu.astype(np.float32), sig.astype(np.float32)


# -----------------------------
# Torch dataset + model
# -----------------------------
class NPArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class MLPBinary(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits shape [B]


@torch.no_grad()
def predict_probs_torch(model: nn.Module, X: np.ndarray, device: str, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    dl = DataLoader(NPArrayDataset(X, np.zeros((X.shape[0],), dtype=np.int64)),
                    batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    for xb, _ in dl:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict:
    y_pred = (probs >= threshold).astype(np.int64)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    out = {
        "threshold": float(threshold),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
    }
    # AUC is undefined if only one class exists in y_true
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, probs))
    else:
        out["roc_auc"] = None
    return out


# -----------------------------
# Training entrypoint
# -----------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    npz_paths = sorted(glob.glob(os.path.join(args.features_dir, "*.npz")))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in {args.features_dir}")

    # Determine feature usage
    use_audio, use_video = choose_feature_flags(args, npz_paths[0])

    holdout_path = os.path.join(args.features_dir, f"{args.holdout_vod}.npz")
    if holdout_path not in npz_paths:
        # still allow holdout even if glob ordering differs
        if not os.path.exists(holdout_path):
            raise FileNotFoundError(
                f"Holdout file not found: {holdout_path}\n"
                f"Existing: {[os.path.basename(p) for p in npz_paths]}"
            )

    # Load all VODs
    train_vods: List[VodData] = []
    test_vod: Optional[VodData] = None

    for p in npz_paths:
        vod_id = os.path.splitext(os.path.basename(p))[0]
        vd = load_npz(p, vod_id=vod_id, want_audio=use_audio, want_video=use_video)
        if vod_id == args.holdout_vod:
            test_vod = vd
        else:
            train_vods.append(vd)

    if test_vod is None:
        # If holdout wasn't in glob, load directly
        test_vod = load_npz(holdout_path, vod_id=args.holdout_vod, want_audio=use_audio, want_video=use_video)

    if not train_vods:
        raise ValueError("No training VODs found (everything is holdout?). Add more .npz or change --holdout_vod.")

    X_train = np.concatenate([v.X for v in train_vods], axis=0)
    y_train = np.concatenate([v.y for v in train_vods], axis=0)
    X_test = test_vod.X
    y_test = test_vod.y

    # Standardize
    X_train_s, X_test_s, mu, sig = standardize_train_test(X_train, X_test)

    # Report class balance
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    print(f"Train segments: {len(y_train)}  positives={pos}  negatives={neg}  pos_rate={pos/len(y_train):.4f}")
    pos_t = int(y_test.sum())
    print(f"Test  segments: {len(y_test)}  positives={pos_t}  pos_rate={pos_t/len(y_test):.4f}")
    print(f"Using features: audio={use_audio}, video={use_video}, dim={X_train_s.shape[1]}")
    print(f"Model: {args.model}")

    metrics: Dict = {
        "holdout_vod": args.holdout_vod,
        "use_audio": use_audio,
        "use_video": use_video,
        "feature_dim": int(X_train_s.shape[1]),
        "train_counts": {"pos": pos, "neg": neg},
        "test_counts": {"pos": pos_t, "neg": int((y_test == 0).sum())},
    }

    if args.model == "logreg":
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn LogisticRegression not available. pip install scikit-learn")
        # class_weight='balanced' helps with imbalance
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None)
        clf.fit(X_train_s, y_train)
        probs = clf.predict_proba(X_test_s)[:, 1]
        m = compute_metrics(y_test, probs, args.threshold)
        metrics["test_metrics"] = m
        print("\nTEST METRICS:", json.dumps(m, indent=2))

        # Save
        ckpt = {
            "model_type": "logreg",
            "sklearn_coef_": clf.coef_.astype(np.float32),
            "sklearn_intercept_": clf.intercept_.astype(np.float32),
            "scaler_mean": mu,
            "scaler_std": sig,
            "config": vars(args),
            "metrics": metrics,
        }
        torch.save(ckpt, args.out_path)
        print(f"\nSaved checkpoint: {args.out_path}")

    else:
        device = args.device
        in_dim = X_train_s.shape[1]
        model = MLPBinary(in_dim=in_dim, hidden=args.hidden, dropout=args.dropout).to(device)

        # Weighted BCE to handle imbalance:
        # pos_weight = neg/pos (PyTorch convention)
        pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_dl = DataLoader(NPArrayDataset(X_train_s, y_train), batch_size=args.batch_size, shuffle=True, num_workers=0)

        # Training loop
        model.train()
        for epoch in range(1, args.epochs + 1):
            losses = []
            for xb, yb in train_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
            avg_loss = float(np.mean(losses)) if losses else 0.0

            # quick eval each epoch (optional)
            probs = predict_probs_torch(model, X_test_s, device=device)
            m = compute_metrics(y_test, probs, args.threshold)
            print(f"Epoch {epoch:02d}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"test_f1={m['f1']:.3f}  prec={m['precision']:.3f}  rec={m['recall']:.3f}")

        # Final metrics
        probs = predict_probs_torch(model, X_test_s, device=device)
        m = compute_metrics(y_test, probs, args.threshold)
        metrics["test_metrics"] = m
        print("\nFINAL TEST METRICS:", json.dumps(m, indent=2))

        # Save checkpoint
        ckpt = {
            "model_type": "mlp",
            "state_dict": model.state_dict(),
            "scaler_mean": mu,
            "scaler_std": sig,
            "config": vars(args),
            "feature_dim": int(in_dim),
            "metrics": metrics,
        }
        torch.save(ckpt, args.out_path)
        print(f"\nSaved checkpoint: {args.out_path}")

    # Optional metrics json
    if args.save_metrics_json:
        os.makedirs(os.path.dirname(args.save_metrics_json) or ".", exist_ok=True)
        with open(args.save_metrics_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics JSON: {args.save_metrics_json}")


if __name__ == "__main__":
    main()
