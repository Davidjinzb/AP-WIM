#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXP_DIR="$ROOT/scorer_audit_experiments"
PY="${APWIM_PYTHON:-python}"
RUN_ID="${RUN_ID:-review_tinyimagenet_retrieval_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/$RUN_ID}"
TASK_OUT="$OUT_ROOT/tinyimagenet_retrieval_ranking"
LOG_DIR="$OUT_ROOT/logs"

mkdir -p "$TASK_OUT" "$LOG_DIR"
export APWIM_WAM_EXPERIMENT_DIR="${APWIM_WAM_EXPERIMENT_DIR:-$ROOT/experiments/wam_experiments}"
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --output_root "$TASK_OUT"
  --tiny_root "${APWIM_TINY_IMAGENET_ROOT:-$ROOT/data/tiny-imagenet-200}"
  --gallery_per_class "${GALLERY_PER_CLASS:-10}"
  --query_per_class "${QUERY_PER_CLASS:-3}"
  --max_classes "${MAX_CLASSES:-200}"
  --attr_batch_size "${ATTR_BATCH_SIZE:-8}"
  --eval_batch_size "${EVAL_BATCH_SIZE:-32}"
  --embed_batch_size "${EMBED_BATCH_SIZE:-64}"
  --fractions "${FRACTIONS:-0,0.05,0.1,0.2,0.3,0.5,0.7,1.0}"
)

(
  cd "$APWIM_WAM_EXPERIMENT_DIR"
  CUDA_VISIBLE_DEVICES="${GPU0_DEVICE:-0}" "$PY" "$EXP_DIR/tinyimagenet_retrieval_ranking_audit.py"     "${COMMON_ARGS[@]}"     --methods ap_wim,wam_ig,ig,fast_ig,saliencymap
) > "$LOG_DIR/retrieval_rank_gpu0.log" 2>&1 &
PID0=$!

(
  cd "$APWIM_WAM_EXPERIMENT_DIR"
  CUDA_VISIBLE_DEVICES="${GPU1_DEVICE:-1}" "$PY" "$EXP_DIR/tinyimagenet_retrieval_ranking_audit.py"     "${COMMON_ARGS[@]}"     --methods mfaba,la,agi,sg,wam_sg
) > "$LOG_DIR/retrieval_rank_gpu1.log" 2>&1 &
PID1=$!

wait "$PID0"
wait "$PID1"

(
  cd "$APWIM_WAM_EXPERIMENT_DIR"
  CUDA_VISIBLE_DEVICES="${GPU0_DEVICE:-0}" "$PY" "$EXP_DIR/tinyimagenet_retrieval_ranking_audit.py"     "${COMMON_ARGS[@]}"     --methods all     --summarize_only
) > "$LOG_DIR/retrieval_rank_summary.log" 2>&1
