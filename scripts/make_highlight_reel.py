#!/usr/bin/env python3
"""
make_highlight_reel.py

Create highlight reel(s) from clips.csv produced by postprocess_intervals.py.

Input clips.csv columns:
  vod_id,clip_start,clip_end,...

Typical flow:
  python scripts/predict.py ... -> data/preds/preds.csv
  python scripts/postprocess_intervals.py ... -> data/output/clips.csv
  python scripts/make_highlight_reel.py \
      --clips_csv data/output/clips.csv \
      --raw_dir data/raw_vods \
      --out_dir data/output \
      --per_vod

Examples:
  # One reel per VOD (recommended)
  python scripts/make_highlight_reel.py \
    --clips_csv data/output/clips.csv \
    --raw_dir data/raw_vods \
    --out_dir data/output \
    --per_vod

  # Single combined reel across all VODs
  python scripts/make_highlight_reel.py \
    --clips_csv data/output/clips.csv \
    --raw_dir data/raw_vods \
    --out_dir data/output \
    --combined_name highlights_all.mp4

By default, we re-encode clips for reliability across cuts/concat.
Use --stream_copy to attempt faster cutting/concat without re-encoding (may fail).
"""

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class Clip:
    vod_id: str
    start: float
    end: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clips_csv", type=str, required=True)
    p.add_argument("--raw_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--video_ext", type=str, default=".mp4")

    # Output mode
    p.add_argument("--per_vod", action="store_true", help="Write one reel per VOD (recommended).")
    p.add_argument("--combined_name", type=str, default="", help="If set, write a single combined reel name.")

    # Encoding options
    p.add_argument("--stream_copy", action="store_true",
                   help="Try to cut/concat with -c copy (faster but less reliable). Default is re-encode.")
    p.add_argument("--crf", type=int, default=23, help="Quality for libx264 if re-encoding (lower = higher quality).")
    p.add_argument("--preset", type=str, default="veryfast", help="x264 preset if re-encoding.")

    # Audio
    p.add_argument("--audio_bitrate", type=str, default="160k")

    # Clip filtering
    p.add_argument("--min_clip_len", type=float, default=0.5, help="Skip clips shorter than this (seconds).")
    p.add_argument("--max_clips_per_vod", type=int, default=0,
                   help="If >0, only take first N clips per VOD (clips_csv should already be sorted).")

    # Temp
    p.add_argument("--tmp_dir", type=str, default="", help="Optional temp dir. Default: out_dir/tmp_reel")
    p.add_argument("--keep_tmp", action="store_true", help="Keep intermediate cut clips and concat lists.")

    return p.parse_args()


def check_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise RuntimeError("ffmpeg not found. Install ffmpeg and ensure it's on PATH.")


def run(cmd: List[str]) -> None:
    # Print cmd nicely for debugging
    print(" ".join([f'"{c}"' if " " in c else c for c in cmd]))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")


def cut_clip(
    in_path: str,
    out_path: str,
    start: float,
    end: float,
    stream_copy: bool,
    crf: int,
    preset: str,
    audio_bitrate: str,
) -> None:
    duration = max(0.0, end - start)
    if duration <= 0:
        return

    if stream_copy:
        # Fast but can produce non-keyframe issues
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", in_path,
            "-c", "copy",
            out_path,
            "-y",
        ]
    else:
        # Reliable: re-encode
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", in_path,
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-movflags", "+faststart",
            out_path,
            "-y",
        ]
    run(cmd)


def concat_clips(list_path: str, out_path: str, stream_copy: bool, crf: int, preset: str, audio_bitrate: str) -> None:
    if stream_copy:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            out_path,
            "-y",
        ]
    else:
        # Re-encode at concat stage (safe even if clips were stream_copy)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-movflags", "+faststart",
            out_path,
            "-y",
        ]
    run(cmd)


def make_concat_list(list_file: str, clip_paths: List[str]) -> None:
    # FFmpeg concat demuxer expects lines: file 'path'
    with open(list_file, "w", encoding="utf-8") as f:
        for p in clip_paths:
            # Normalize to forward slashes; quoting handles spaces
            f.write(f"file '{p.replace(os.sep, '/')}'\n")


def main() -> None:
    args = parse_args()
    check_ffmpeg()
    os.makedirs(args.out_dir, exist_ok=True)

    tmp_dir = args.tmp_dir.strip() or os.path.join(args.out_dir, "tmp_reel")
    os.makedirs(tmp_dir, exist_ok=True)

    df = pd.read_csv(args.clips_csv)
    if df.empty:
        print(f"No clips found in {args.clips_csv}. Nothing to do.")
        return

    required = {"vod_id", "clip_start", "clip_end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.clips_csv} missing columns: {sorted(missing)}")

    # Sort for stable output
    df = df.sort_values(["vod_id", "clip_start"])

    # Gather clips
    clips: List[Clip] = []
    for r in df.itertuples(index=False):
        start = float(getattr(r, "clip_start"))
        end = float(getattr(r, "clip_end"))
        if end - start < args.min_clip_len:
            continue
        clips.append(Clip(vod_id=str(getattr(r, "vod_id")), start=start, end=end))

    if not clips:
        print("All clips were filtered out (too short).")
        return

    # Determine output mode
    if args.combined_name and args.per_vod:
        raise ValueError("Choose only one: --per_vod OR --combined_name")

    if not args.combined_name and not args.per_vod:
        # default to per_vod for safety
        args.per_vod = True

    # Group by VOD
    by_vod = {}
    for c in clips:
        by_vod.setdefault(c.vod_id, []).append(c)

    outputs = []

    def process_one_reel(vod_ids: List[str], out_name: str) -> None:
        nonlocal outputs
        clip_paths = []
        clip_count = 0

        # Create a subfolder for this reel’s clips (avoids filename collisions)
        reel_tmp = os.path.join(tmp_dir, os.path.splitext(out_name)[0])
        os.makedirs(reel_tmp, exist_ok=True)

        for vid in vod_ids:
            in_path = os.path.join(args.raw_dir, f"{vid}{args.video_ext}")
            if not os.path.exists(in_path):
                print(f"[WARN] Missing video file: {in_path} (skipping VOD {vid})", file=sys.stderr)
                continue

            vod_clips = by_vod.get(vid, [])
            if args.max_clips_per_vod and args.max_clips_per_vod > 0:
                vod_clips = vod_clips[: args.max_clips_per_vod]

            for i, c in enumerate(vod_clips, start=1):
                clip_out = os.path.join(reel_tmp, f"{vid}_clip_{i:04d}.mp4")
                try:
                    cut_clip(
                        in_path=in_path,
                        out_path=clip_out,
                        start=c.start,
                        end=c.end,
                        stream_copy=args.stream_copy,
                        crf=args.crf,
                        preset=args.preset,
                        audio_bitrate=args.audio_bitrate,
                    )
                except Exception as e:
                    print(f"[WARN] Failed cutting {vid} {c.start}-{c.end}: {e}", file=sys.stderr)
                    continue
                clip_paths.append(os.path.abspath(clip_out))
                clip_count += 1

        if clip_count == 0:
            print(f"[WARN] No clips cut for reel {out_name}.")
            return

        list_path = os.path.join(reel_tmp, "concat_list.txt")
        make_concat_list(list_path, clip_paths)

        out_path = os.path.join(args.out_dir, out_name)
        concat_clips(
            list_path=list_path,
            out_path=out_path,
            stream_copy=args.stream_copy,
            crf=args.crf,
            preset=args.preset,
            audio_bitrate=args.audio_bitrate,
        )
        outputs.append(out_path)
        print(f"[OK] Wrote reel: {out_path}")

        if not args.keep_tmp:
            # Remove clip files and concat list
            try:
                shutil.rmtree(reel_tmp)
            except Exception:
                pass

    if args.per_vod:
        for vid in sorted(by_vod.keys()):
            out_name = f"{vid}_highlights.mp4"
            process_one_reel([vid], out_name)
    else:
        # Combined reel across all VODs, in vod_id order
        process_one_reel(sorted(by_vod.keys()), args.combined_name)

    print("\nDone.")
    if outputs:
        print("Outputs:")
        for p in outputs:
            print(" -", p)
    else:
        print("No outputs produced (check warnings above).")


if __name__ == "__main__":
    main()
