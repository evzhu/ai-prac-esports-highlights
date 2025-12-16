#!/usr/bin/env python3
"""
predict.py

Run inference on feature .npz files produced by extract_features.py, using a model.pt
saved by train_classifier.py.

Outputs a CSV with: vod_id,start_sec,end_sec,prob

Examples:
  # Predict on all feature files:
  python scripts/predict.py \
    --model_path data/output/model.pt \
    --features_dir data/features \
    --out_csv data/preds/preds.csv

  # Predict on a single VOD:
  python scripts/predict.py \
    --model_path data/output/model.pt \
    --features_dir data/features \
    --vod_id vod_001 \
    --out_csv data/preds/vod_001_preds.csv
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# -----------------------------
# Models (must match training)
# -----------------------------
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
        return self.net(x).squeeze(-1)  # logits [B]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--features_dir", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)

    p.add_argument("--vod_id", type=str, default="", help="Optional: only predict for this VOD id (e.g., vod_001)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=4096)

    return p.parse_args()


def load_checkpoint(path: str) -> Dict:
    # PyTorch 2.6+ defaults weights_only=True, which breaks checkpoints containing numpy arrays.
    # Safe here because this checkpoint was generated locally by our train script.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_type" not in ckpt:
        raise ValueError(f"{path} doesn't look like a checkpoint from train_classifier.py (missing model_type).")
    if "scaler_mean" not in ckpt or "scaler_std" not in ckpt:
        raise ValueError(f"{path} missing scaler_mean/scaler_std.")
    if "config" not in ckpt:
        raise ValueError(f"{path} missing config.")
    return ckpt


def decide_feature_usage(ckpt: Dict) -> Tuple[bool, bool]:
    """
    Determine whether to use audio/video features based on training-time config.
    """
    cfg = ckpt.get("config", {})
    # train script stored: audio_only/video_only/use_audio/use_video
    if cfg.get("audio_only", False):
        return True, False
    if cfg.get("video_only", False):
        return False, True

    # If user explicitly set these at training time, use them.
    use_audio = bool(cfg.get("use_audio", False))
    use_video = bool(cfg.get("use_video", False))

    # If neither was explicitly set, train script auto-detected. We stored those too in metrics.
    # Prefer metrics if present.
    metrics = ckpt.get("metrics", {})
    if "use_audio" in metrics and "use_video" in metrics:
        return bool(metrics["use_audio"]), bool(metrics["use_video"])

    # Fall back to config flags (may be both False if auto-detection occurred and wasn't stored)
    if not (use_audio or use_video):
        # safest fallback: try both (if present per-file)
        return True, True
    return use_audio, use_video


def build_X_from_npz(npz_path: str, use_audio: bool, use_video: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X, start_sec, end_sec) for a single VOD.
    """
    d = np.load(npz_path)

    # Timestamps
    if "start_sec" in d.files:
        start_sec = d["start_sec"].astype(np.float32)
    else:
        start_sec = np.zeros((d["y"].shape[0],), dtype=np.float32) if "y" in d.files else None

    if "end_sec" in d.files:
        end_sec = d["end_sec"].astype(np.float32)
    else:
        end_sec = np.zeros((d["y"].shape[0],), dtype=np.float32) if "y" in d.files else None

    parts: List[np.ndarray] = []

    # If the checkpoint requests audio/video, require them.
    # If both requested (fallback), include whichever exists.
    if use_video and "X_video" in d.files:
        parts.append(d["X_video"].astype(np.float32))
    elif use_video and "X_video" not in d.files and not (use_audio and "X_audio" in d.files):
        raise ValueError(f"{npz_path}: checkpoint expects video features but X_video not found.")

    if use_audio and "X_audio" in d.files:
        parts.append(d["X_audio"].astype(np.float32))
    elif use_audio and "X_audio" not in d.files and not (use_video and "X_video" in d.files):
        raise ValueError(f"{npz_path}: checkpoint expects audio features but X_audio not found.")

    if not parts:
        raise ValueError(f"{npz_path}: no selected features found (use_audio={use_audio}, use_video={use_video}).")

    X = np.concatenate(parts, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if start_sec is None or end_sec is None:
        raise ValueError(f"{npz_path}: missing start_sec/end_sec. Ensure extract_features saved them.")
    return X, start_sec, end_sec


def standardize(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    mu = mu.reshape(1, -1)
    sig = sig.reshape(1, -1)
    sig = np.where(sig < 1e-8, 1.0, sig)
    Xs = (X - mu) / sig
    return Xs.astype(np.float32)


@torch.no_grad()
def predict_probs_mlp(model: nn.Module, X: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    probs = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def predict_probs_logreg(ckpt: Dict, X: np.ndarray) -> np.ndarray:
    """
    For logreg checkpoint saved by our train script:
      prob = sigmoid(X @ coef.T + intercept)
    """
    coef = ckpt["sklearn_coef_"].astype(np.float32)      # [1, D]
    intercept = ckpt["sklearn_intercept_"].astype(np.float32)  # [1]
    z = X @ coef.T + intercept  # [N, 1]
    z = z.squeeze(1)
    probs = 1.0 / (1.0 + np.exp(-z))
    return probs.astype(np.float32)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    ckpt = load_checkpoint(args.model_path)
    model_type = ckpt["model_type"]
    use_audio, use_video = decide_feature_usage(ckpt)

    mu = ckpt["scaler_mean"].astype(np.float32).reshape(-1)
    sig = ckpt["scaler_std"].astype(np.float32).reshape(-1)

    # Gather npz paths
    if args.vod_id:
        npz_paths = [os.path.join(args.features_dir, f"{args.vod_id}.npz")]
        if not os.path.exists(npz_paths[0]):
            raise FileNotFoundError(f"Could not find {npz_paths[0]}")
    else:
        npz_paths = sorted(glob.glob(os.path.join(args.features_dir, "*.npz")))
        if not npz_paths:
            raise FileNotFoundError(f"No .npz found in {args.features_dir}")

    # Build model if needed
    device = args.device
    mlp_model: Optional[nn.Module] = None
    if model_type == "mlp":
        in_dim = int(ckpt.get("feature_dim", mu.shape[0]))
        cfg = ckpt.get("config", {})
        hidden = int(cfg.get("hidden", 256))
        dropout = float(cfg.get("dropout", 0.2))
        mlp_model = MLPBinary(in_dim=in_dim, hidden=hidden, dropout=dropout).to(device)
        mlp_model.load_state_dict(ckpt["state_dict"])
        mlp_model.eval()
    elif model_type == "logreg":
        # no torch model needed
        pass
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")

    rows = []
    for npz_path in npz_paths:
        vod_id = os.path.splitext(os.path.basename(npz_path))[0]
        try:
            X, start_sec, end_sec = build_X_from_npz(npz_path, use_audio=use_audio, use_video=use_video)
        except Exception as e:
            print(f"[ERROR] {vod_id}: {e}", file=sys.stderr)
            continue

        # Standardize
        if X.shape[1] != mu.shape[0]:
            raise ValueError(
                f"{vod_id}: feature dim mismatch. X has dim {X.shape[1]} but scaler has dim {mu.shape[0]}.\n"
                f"Did you train with different feature settings (audio/video) than you extracted?"
            )
        Xs = standardize(X, mu, sig)

        # Predict
        if model_type == "mlp":
            probs = predict_probs_mlp(mlp_model, Xs, device=device, batch_size=args.batch_size)
        else:
            probs = predict_probs_logreg(ckpt, Xs)

        for s, e, p in zip(start_sec, end_sec, probs):
            rows.append((vod_id, float(s), float(e), float(p)))

        print(f"[OK] Predicted {len(probs)} segments for {vod_id}")

    df = pd.DataFrame(rows, columns=["vod_id", "start_sec", "end_sec", "prob"])
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote predictions: {args.out_csv}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
