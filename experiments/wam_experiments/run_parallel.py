# #!/usr/bin/env bash
# set -e

# # ====== user config ======
# GPUS=(0 1)                 # 2 张卡
# MAX_PER_GPU=3              # 每张卡同时跑 2~3 个：改成 2 或 3
# EPOCHS=50
# BATCH=32
# WORKERS=4                  # 并行多进程时，workers 建议别太大，避免 IO/CPU 打架
# LR=1e-3
# WD=0.0
# FOLD=1
# SEED=42
# PRETRAINED=0               # 1=使用 torchvision 预训练; 0=不使用
# AMP=1                      # 1=amp; 0=不用amp
# OUTROOT="runs_esc50"
# LOGDIR="logs_esc50"

# # 10~15 个模型（你可以增删）
# MODELS=(
#   resnet18
#   resnet34
#   resnet50
#   mobilenet_v2
#   mobilenet_v3_small
#   mobilenet_v3_large
#   efficientnet_b0
#   efficientnet_b1
#   densenet121
#   shufflenet_v2_x1_0
#   squeezenet1_1
#   convnext_tiny
#   convnext_small
#   vit_b_16
#   regnet_y_400mf
# )

# mkdir -p "$LOGDIR"
# mkdir -p "$OUTROOT"

# # ====== internal scheduler ======
# declare -A PID2GPU
# PIDS=()

# running_on_gpu() {
#   local gpu="$1"
#   local cnt=0
#   for pid in "${PIDS[@]}"; do
#     if kill -0 "$pid" 2>/dev/null; then
#       if [[ "${PID2GPU[$pid]}" == "$gpu" ]]; then
#         cnt=$((cnt+1))
#       fi
#     fi
#   done
#   echo "$cnt"
# }

# pick_gpu_with_slot() {
#   # 找一个还有空位的 gpu；如果都满了就返回空
#   for gpu in "${GPUS[@]}"; do
#     local cnt
#     cnt="$(running_on_gpu "$gpu")"
#     if [[ "$cnt" -lt "$MAX_PER_GPU" ]]; then
#       echo "$gpu"
#       return
#     fi
#   done
#   echo ""
# }

# cleanup_finished() {
#   # 清理已经结束的 pid
#   local new_pids=()
#   for pid in "${PIDS[@]}"; do
#     if kill -0 "$pid" 2>/dev/null; then
#       new_pids+=("$pid")
#     else
#       unset PID2GPU["$pid"]
#     fi
#   done
#   PIDS=("${new_pids[@]}")
# }

# launch_one() {
#   local model="$1"
#   local gpu="$2"

#   local tag="${model}_gpu${gpu}"
#   local logfile="${LOGDIR}/${tag}.log"

#   echo "[LAUNCH] model=${model} on GPU=${gpu} -> ${logfile}"

#   # 关键：CUDA_VISIBLE_DEVICES=$gpu -> 训练脚本里看到的是 cuda:0
#   CUDA_VISIBLE_DEVICES="$gpu" \
#   OMP_NUM_THREADS=1 \
#   MKL_NUM_THREADS=1 \
#   python train_esc50.py \
#     --model "$model" \
#     --fold "$FOLD" \
#     --epochs "$EPOCHS" \
#     --batch-size "$BATCH" \
#     --num-workers "$WORKERS" \
#     --lr "$LR" \
#     --weight-decay "$WD" \
#     --seed "$SEED" \
#     --outdir "$OUTROOT" \
#     $( [[ "$PRETRAINED" -eq 1 ]] && echo "--pretrained" ) \
#     $( [[ "$AMP" -eq 1 ]] && echo "--amp" ) \
#     > "$logfile" 2>&1 &

#   local pid=$!
#   PIDS+=("$pid")
#   PID2GPU["$pid"]="$gpu"
# }

# # ====== main loop ======
# for model in "${MODELS[@]}"; do
#   while true; do
#     cleanup_finished
#     gpu="$(pick_gpu_with_slot)"
#     if [[ -n "$gpu" ]]; then
#       launch_one "$model" "$gpu"
#       break
#     fi
#     sleep 2
#   done
# done

# # wait all
# echo "[WAIT] all jobs launched, waiting..."
# wait
# echo "[DONE] all trainings finished."

import multiprocessing
pool = multiprocessing.Pool(processes=6)
all_models = [
      'resnet18',
  'resnet34',
  'resnet50',
  'mobilenet_v2',
  'mobilenet_v3_small',
  'mobilenet_v3_large',
  'efficientnet_b0',
  'efficientnet_b1',
  'densenet121',
  'shufflenet_v2_x1_0',
  'squeezenet1_1',
  'convnext_tiny',
  'convnext_small',
  'vit_b_16',
]
all_commands = []
cuda_flag = False
for model in all_models:
    cmd = f"CUDA_VISIBLE_DEVICES={int(cuda_flag)} python train_esc50.py --model {model} --epochs 50 --batch-size 32 --num-workers 4 --lr 1e-3 --weight-decay 0.0 --outdir runs_esc50 --pretrained --amp"
    all_commands.append(cmd)
    cuda_flag = not cuda_flag  # Alternate between using and not using CUDA

import os
pool.imap(os.system, all_commands)
pool.close()
pool.join()