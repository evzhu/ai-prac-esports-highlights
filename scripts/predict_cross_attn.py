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

from __future__ import annotations

import argparse
import csv
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class MLPBinary(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CrossModalFusion(nn.Module):
    def __init__(self, dv: int, da: int, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.v_proj = nn.Linear(dv, d_model)
        self.a_proj = nn.Linear(da, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, V: torch.Tensor, A: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        V = self.v_proj(V)
        A = self.a_proj(A)
        Z, _ = self.cross_attn(query=V, key=A, value=A, key_padding_mask=key_padding_mask)
        X = self.ln1(V + Z)
        X = self.ln2(X + self.ffn(X))
        return self.head(X).squeeze(-1)  # [B, L]


def standardize_with_stats(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    sig = np.where(sig < 1e-8, 1.0, sig)
    return ((X - mu) / sig).astype(np.float32)


def build_X_from_npz(npz_path: str, use_audio: bool, use_video: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X, start_sec, end_sec) for a single VOD (concat features).
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

    if use_video and "X_video" in d.files:
        parts.append(d["X_video"].astype(np.float32))
    elif use_video and "X_video" not in d.files and not (use_audio and "X_audio" in d.files):
        raise ValueError(f"{npz_path}: checkpoint expects video features but X_video not found.")

    if use_audio and "X_audio" in d.files:
        parts.append(d["X_audio"].astype(np.float32))
    elif use_audio and "X_audio" not in d.files and not (use_video and "X_video" in d.files):
        raise ValueError(f"{npz_path}: checkpoint expects audio features but X_audio not found.")

    if not parts:
        raise ValueError(f"{npz_path}: no features found/selected.")

    X = np.concatenate(parts, axis=1).astype(np.float32)
    return X, start_sec, end_sec


def build_VA_from_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (V, A, start_sec, end_sec) for a single VOD.
    Requires both X_video and X_audio.
    """
    d = np.load(npz_path)
    if "X_video" not in d.files or "X_audio" not in d.files:
        raise ValueError(f"{npz_path}: cross_attn requires BOTH X_video and X_audio.")
    V = d["X_video"].astype(np.float32)
    A = d["X_audio"].astype(np.float32)

    if "start_sec" in d.files:
        start_sec = d["start_sec"].astype(np.float32)
    else:
        start_sec = np.zeros((V.shape[0],), dtype=np.float32)

    if "end_sec" in d.files:
        end_sec = d["end_sec"].astype(np.float32)
    else:
        end_sec = np.zeros((V.shape[0],), dtype=np.float32)

    return V, A, start_sec, end_sec


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Path to model checkpoint saved by train_classifier.py")
    ap.add_argument("--features_dir", required=True, help="Directory containing per-VOD .npz feature files")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--vod_id", default=None, help="If set, only run inference on this VOD (e.g., vod_003)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model_type = ckpt.get("model_type", "mlp")

    mu = ckpt.get("feature_mean", None)
    sig = ckpt.get("feature_std", None)
    if mu is not None:
        mu = np.asarray(mu, dtype=np.float32).reshape(1, -1)
    if sig is not None:
        sig = np.asarray(sig, dtype=np.float32).reshape(1, -1)

    device = args.device

    # Gather npz paths
    if args.vod_id:
        npz_paths = [os.path.join(args.features_dir, f"{args.vod_id}.npz")]
    else:
        npz_paths = sorted(
            [os.path.join(args.features_dir, f) for f in os.listdir(args.features_dir) if f.endswith(".npz")]
        )

    # Prepare model
    torch_model: Optional[nn.Module] = None
    sk_model = None

    if model_type == "logreg":
        sk_model = ckpt["sk_model"]
        use_audio = bool(ckpt.get("config", {}).get("use_audio", True))
        use_video = bool(ckpt.get("config", {}).get("use_video", True))
    elif model_type == "mlp":
        cfg = ckpt.get("config", {})
        in_dim = int(ckpt.get("feature_dim"))
        hidden = int(cfg.get("hidden", 256))
        dropout = float(cfg.get("dropout", 0.2))
        use_audio = bool(cfg.get("use_audio", True))
        use_video = bool(cfg.get("use_video", True))
        torch_model = MLPBinary(in_dim=in_dim, hidden=hidden, dropout=dropout).to(device)
        torch_model.load_state_dict(ckpt["state_dict"])
        torch_model.eval()
    elif model_type == "cross_attn":
        cfg = ckpt.get("config", {})
        dv = int(cfg.get("dv"))
        da = int(cfg.get("da"))
        d_model = int(cfg.get("d_model", 256))
        n_heads = int(cfg.get("n_heads", 4))
        dropout = float(cfg.get("dropout", 0.2))
        torch_model = CrossModalFusion(dv=dv, da=da, d_model=d_model, n_heads=n_heads, dropout=dropout).to(device)
        torch_model.load_state_dict(ckpt["state_dict"])
        torch_model.eval()
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")

    # Write output rows
    rows = []
    for npz_path in npz_paths:
        vod_id = os.path.splitext(os.path.basename(npz_path))[0]

        if model_type in ("mlp", "logreg"):
            X, start_sec, end_sec = build_X_from_npz(npz_path, use_audio=use_audio, use_video=use_video)
            if mu is not None and sig is not None:
                X = standardize_with_stats(X, mu, sig)

            if model_type == "logreg":
                probs = sk_model.predict_proba(X)[:, 1].astype(np.float32)
            else:
                xb = torch.from_numpy(X).to(device)
                with torch.no_grad():
                    logits = torch_model(xb).detach().cpu().numpy()
                probs = sigmoid(logits).astype(np.float32)

        else:
            # cross_attn
            V, A, start_sec, end_sec = build_VA_from_npz(npz_path)
            # standardize using concat stats
            if mu is not None and sig is not None:
                Xc = np.concatenate([V, A], axis=1)
                Xcs = standardize_with_stats(Xc, mu, sig)
                dv = int(V.shape[1])
                V = Xcs[:, :dv]
                A = Xcs[:, dv:]

            Vt = torch.from_numpy(V).unsqueeze(0).to(device)  # [1, N, Dv]
            At = torch.from_numpy(A).unsqueeze(0).to(device)  # [1, N, Da]
            with torch.no_grad():
                logits = torch_model(Vt, At).squeeze(0).detach().cpu().numpy()
            probs = sigmoid(logits).astype(np.float32)

        for s, e, p in zip(start_sec, end_sec, probs):
            rows.append([vod_id, float(s), float(e), float(p)])

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vod_id", "start_sec", "end_sec", "prob"])
        w.writerows(rows)

    print(f"Wrote predictions: {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
