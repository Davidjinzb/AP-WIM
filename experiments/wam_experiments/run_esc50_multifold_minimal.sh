#!/usr/bin/env bash
set -euo pipefail

cd experiments/wam_experiments

WAM_PYTHON="${WAM_PYTHON:-python}"
RUN_ROOT="${APWIM_MULTIFOLD_RUN_ROOT:-runs/esc50_multifold_$(date +%Y%m%d_%H%M%S)}"
ATTR_DIR="$RUN_ROOT/attributions"
SCORE_DIR="$RUN_ROOT/scores"
LOG_DIR="$RUN_ROOT/logs"
TABLE_DIR="$RUN_ROOT/tables"
CKPT_ROOT="${APWIM_ESC50_CKPT_ROOT:-runs/E1_imagenet_20260509_200917/runs_esc50}"

export APWIM_ESC50_ROOT="${APWIM_ESC50_ROOT:-data/ESC-50-master}"

METHODS="${APWIM_ESC50_METHODS:-ap_wim wam_ig ig}"
FOLDS="${APWIM_ESC50_FOLDS:-1 2 3 4 5}"
ATTR_BATCH_SIZE="${APWIM_ATTR_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${APWIM_EVAL_BATCH_SIZE:-64}"
ROAD_BATCH_SIZE="${APWIM_ROAD_BATCH_SIZE:-64}"

mkdir -p "$ATTR_DIR" "$SCORE_DIR" "$LOG_DIR" "$TABLE_DIR"

run_model() {
  local model="$1"
  local gpu="$2"
  for fold in $FOLDS; do
    local tag="fold${fold}"
    for method in $METHODS; do
      echo "[START] model=$model fold=$fold method=$method gpu=$gpu"
      CUDA_VISIBLE_DEVICES="$gpu" "$WAM_PYTHON" -u generate_attribution_fold.py \
        --model "$model" \
        --fold "$fold" \
        --attr_method "$method" \
        --batch_size "$ATTR_BATCH_SIZE" \
        --attribution_dir "$ATTR_DIR" \
        --output_tag "$tag" \
        --ckpt_root "$CKPT_ROOT" \
        2>&1 | tee "$LOG_DIR/attr_${model}_${method}_${tag}.log"

      CUDA_VISIBLE_DEVICES="$gpu" "$WAM_PYTHON" -u eval_fold.py \
        --model "$model" \
        --fold "$fold" \
        --attr_method "$method" \
        --batch_size "$EVAL_BATCH_SIZE" \
        --attribution_dir "$ATTR_DIR" \
        --score_dir "$SCORE_DIR" \
        --output_tag "$tag" \
        --ckpt_root "$CKPT_ROOT" \
        2>&1 | tee "$LOG_DIR/causal_${model}_${method}_${tag}.log"

      CUDA_VISIBLE_DEVICES="$gpu" "$WAM_PYTHON" -u test_road_fold.py \
        --model "$model" \
        --fold "$fold" \
        --attr_method "$method" \
        --batch_size "$ROAD_BATCH_SIZE" \
        --attribution_dir "$ATTR_DIR" \
        --score_dir "$SCORE_DIR" \
        --output_tag "$tag" \
        --ckpt_root "$CKPT_ROOT" \
        --baseline blur \
        --target pred \
        2>&1 | tee "$LOG_DIR/road_${model}_${method}_${tag}.log"
      echo "[DONE] model=$model fold=$fold method=$method"
    done
  done
}

run_model resnet50 0 &
pid_resnet=$!
run_model mobilenet_v2 1 &
pid_mobilenet=$!

wait "$pid_resnet"
wait "$pid_mobilenet"

METHODS_CSV="$(echo "$METHODS" | tr ' ' ',')"
FOLDS_CSV="$(echo "$FOLDS" | tr ' ' ',')"
"$WAM_PYTHON" -u summarize_esc50_multifold.py \
  --score_dir "$SCORE_DIR" \
  --out_dir "$TABLE_DIR" \
  --methods "$METHODS_CSV" \
  --folds "$FOLDS_CSV" \
  --models "resnet50,mobilenet_v2" \
  2>&1 | tee "$LOG_DIR/summarize.log"

echo "[ALL DONE] $RUN_ROOT"
