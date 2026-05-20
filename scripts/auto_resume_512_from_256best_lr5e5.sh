#!/usr/bin/env bash

set -u
set -o pipefail

PROJECT_DIR="/mnt/disk1/zwh/Project/PanSplat"
cd "${PROJECT_DIR}" || exit 1

GPU_ID=7
RUN_ID="p512_geo_concat_from_256best_lr5e5_auto"
EXPERIMENT="pansplat-512-mix-spe-mona-multi-wsmse-geo-concat"
MAX_EPOCHS=10

INITIAL_CKPT="/mnt/disk1/zwh/Project/PanSplat/logs/hk0l83e3/checkpoints/best-19-101439.ckpt"

RUN_DIR="${PROJECT_DIR}/logs/${RUN_ID}"
CKPT_DIR="${RUN_DIR}/checkpoints"
LATEST_EPOCH_CKPT="${CKPT_DIR}/latest-epoch.ckpt"
AUTO_LOG="${RUN_DIR}/auto_resume.log"

MAX_RESTARTS=20
SLEEP_SECONDS=10

MODEL_VIEW_BATCH=1
LOSS_VIEW_BATCH=1
LR="5.e-5"

mkdir -p "${RUN_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 避免 W&B 在线认证/代理问题。注意这里还会在 python 命令里显式传 wandb.mode=offline。
export WANDB_MODE=offline

restart_count=0

while true; do
  echo "============================================================" | tee -a "${AUTO_LOG}"
  echo "[auto-resume] Start time: $(date)" | tee -a "${AUTO_LOG}"
  echo "[auto-resume] Restart count: ${restart_count}/${MAX_RESTARTS}" | tee -a "${AUTO_LOG}"
  echo "[auto-resume] RUN_ID: ${RUN_ID}" | tee -a "${AUTO_LOG}"
  echo "============================================================" | tee -a "${AUTO_LOG}"

  if [ -f "${LATEST_EPOCH_CKPT}" ]; then
    echo "[auto-resume] Resume from latest epoch checkpoint:" | tee -a "${AUTO_LOG}"
    echo "[auto-resume] ${LATEST_EPOCH_CKPT}" | tee -a "${AUTO_LOG}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    WANDB_RUN_ID=${RUN_ID} \
    python -m src.main \
      +experiment=${EXPERIMENT} \
      checkpointing.load=${LATEST_EPOCH_CKPT} \
      model.encoder.geometry_reliability.return_debug=false \
      model.decoder.view_batch=${MODEL_VIEW_BATCH} \
      loss.pyimage.decoder.view_batch=${LOSS_VIEW_BATCH} \
      optimizer.lr=${LR} \
      trainer.max_epochs=${MAX_EPOCHS} \
      checkpointing.save_latest_epoch=true \
      wandb.mode=offline \
      2>&1 | tee -a "${AUTO_LOG}"

    exit_code=${PIPESTATUS[0]}

  else
    echo "[auto-resume] latest-epoch.ckpt not found." | tee -a "${AUTO_LOG}"
    echo "[auto-resume] Start 512 fine-tuning from 256 best weights:" | tee -a "${AUTO_LOG}"
    echo "[auto-resume] ${INITIAL_CKPT}" | tee -a "${AUTO_LOG}"

    if [ ! -f "${INITIAL_CKPT}" ]; then
      echo "[auto-resume] ERROR: INITIAL_CKPT not found: ${INITIAL_CKPT}" | tee -a "${AUTO_LOG}"
      exit 1
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    WANDB_RUN_ID=${RUN_ID} \
    python -m src.main \
      +experiment=${EXPERIMENT} \
      model.weights_path=${INITIAL_CKPT} \
      model.encoder.geometry_reliability.return_debug=false \
      model.decoder.view_batch=${MODEL_VIEW_BATCH} \
      loss.pyimage.decoder.view_batch=${LOSS_VIEW_BATCH} \
      optimizer.lr=${LR} \
      trainer.max_epochs=${MAX_EPOCHS} \
      checkpointing.save_latest_epoch=true \
      wandb.mode=offline \
      2>&1 | tee -a "${AUTO_LOG}"

    exit_code=${PIPESTATUS[0]}
  fi

  echo "============================================================" | tee -a "${AUTO_LOG}"
  echo "[auto-resume] Process exited at: $(date)" | tee -a "${AUTO_LOG}"
  echo "[auto-resume] Exit code: ${exit_code}" | tee -a "${AUTO_LOG}"
  echo "============================================================" | tee -a "${AUTO_LOG}"

  if [ "${exit_code}" -eq 0 ]; then
    echo "[auto-resume] Training finished normally. Stop." | tee -a "${AUTO_LOG}"
    exit 0
  fi

  if [ "${exit_code}" -eq 130 ] || [ "${exit_code}" -eq 143 ]; then
    echo "[auto-resume] Interrupted manually. Stop." | tee -a "${AUTO_LOG}"
    exit "${exit_code}"
  fi

  restart_count=$((restart_count + 1))

  if [ "${restart_count}" -ge "${MAX_RESTARTS}" ]; then
    echo "[auto-resume] ERROR: reached max restarts: ${MAX_RESTARTS}" | tee -a "${AUTO_LOG}"
    exit "${exit_code}"
  fi

  echo "[auto-resume] Training crashed. Restart after ${SLEEP_SECONDS}s..." | tee -a "${AUTO_LOG}"
  sleep "${SLEEP_SECONDS}"
done
