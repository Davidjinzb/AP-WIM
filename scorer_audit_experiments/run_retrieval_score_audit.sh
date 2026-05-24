#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WAM_DIR="${APWIM_WAM_EXPERIMENT_DIR:-$PROJECT_ROOT/experiments/wam_experiments}"
WAM_PYTHON="${WAM_PYTHON:-python}"
RUN_ID="${RUN_ID:-review_audit_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/results/$RUN_ID}"
LOG_DIR="$OUT_ROOT/logs"
SCRIPT_DIR="$PROJECT_ROOT/scorer_audit_experiments"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export APWIM_WAM_EXPERIMENT_DIR="$WAM_DIR"

mkdir -p "$OUT_ROOT"/{logs,tables,attributions,scores}
cd "$WAM_DIR"

echo "[INFO] project=$PROJECT_ROOT"
echo "[INFO] out=$OUT_ROOT"
echo "[INFO] python=$WAM_PYTHON"
echo "[INFO] scripts=$SCRIPT_DIR"

GPU0_METHODS="${GPU0_METHODS:-ap_wim,wam_ig,ig,fast_ig,saliencymap}"
GPU1_METHODS="${GPU1_METHODS:-mfaba,la,agi,sg,wam_sg}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"

CUDA_VISIBLE_DEVICES="${GPU0_DEVICE:-0}" "$WAM_PYTHON" "$SCRIPT_DIR/retrieval_score_audit.py"   --output_root "$OUT_ROOT/retrieval_score_audit"   --methods "$GPU0_METHODS"   --max_samples "$MAX_SAMPLES"   --attr_batch_size "${ATTR_BATCH_SIZE:-8}"   --eval_batch_size "${EVAL_BATCH_SIZE:-32}"   > "$LOG_DIR/retrieval_gpu0.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES="${GPU1_DEVICE:-1}" "$WAM_PYTHON" "$SCRIPT_DIR/retrieval_score_audit.py"   --output_root "$OUT_ROOT/retrieval_score_audit"   --methods "$GPU1_METHODS"   --max_samples "$MAX_SAMPLES"   --attr_batch_size "${ATTR_BATCH_SIZE:-8}"   --eval_batch_size "${EVAL_BATCH_SIZE:-32}"   > "$LOG_DIR/retrieval_gpu1.log" 2>&1 &
PID1=$!

wait "$PID0"
wait "$PID1"

ALL_METHODS="${ALL_METHODS:-ap_wim,wam_ig,ig,fast_ig,saliencymap,mfaba,la,agi,sg,wam_sg}"
"$WAM_PYTHON" "$SCRIPT_DIR/retrieval_score_audit.py"   --output_root "$OUT_ROOT/retrieval_score_audit"   --methods "$ALL_METHODS"   --max_samples "$MAX_SAMPLES"   --summarize_only   > "$LOG_DIR/retrieval_summary.log" 2>&1
