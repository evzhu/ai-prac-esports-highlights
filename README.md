# AI Prac: Esports Highlighter Project

## Project Structure
Organized into data folder - with annotations, raw esports videos, metadata for the videos, and the split segments. Scripts folder - stores scripts for filling in the annotation document, segmenting videos, etc.

# Notes for Dev
Install FFmpeg system-wide. Do not run the segment script or annotation script inside your venv.

# Base Model
py scripts/predict.py `
  --model_path data/models/model.pt `
  --features_dir data/features `
  --vod_id vod_003 `
  --out_csv data/output/vod_003_preds.csv

py scripts/postprocess_intervals.py `
  --preds_csv data/output/vod_003_preds.csv `
  --out_csv data/output/vod_003_clips.csv

py scripts/make_highlight_reel.py `
  --clips_csv data/output/vod_003_clips.csv `
  --raw_dir data/raw_vods `
  --out_path data/output/vod_003_highlights.mp4


# Cross-Attention
py scripts/predict_cross_attn.py `
  --model_path data/models/model_cross_attn.pt `
  --features_dir data/features `
  --vod_id vod_003 `
  --out_csv data/cross_attn/vod_003_preds.csv `
  --device cpu
py scripts/postprocess_intervals.py `
  --preds_csv data/cross_attn/vod_003_preds.csv `
  --out_csv data/cross_attn/vod_003_clips.csv
py scripts/make_highlight_reel.py `
  --clips_csv data/cross_attn/vod_003_clips.csv `
  --raw_dir data/raw_vods `
  --out_path data/cross_attn/vod_003_highlights.mp4
