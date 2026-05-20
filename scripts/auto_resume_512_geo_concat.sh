#!/usr/bin/env bash

set -u
set -o pipefail

# ============================================================
# Auto-resume script for 512 geo-concat training.
#
# First run:
#   Use INITIAL_CKPT as model.weights_path.
#
# After latest-epoch.ckpt exists:
#   Use latest-epoch.ckpt as checkpointing.load to resume full state.
# ============================================================

PROJECT_DIR="/mnt/disk1/zwh/Project/PanSplat"
cd "${PROJECT_DIR}" || exit 1

# ----------------------------
# Basic settings
# ----------------------------
GPU_ID=6

# Fixed run id. Do not leave this random for auto-resume,
# because the script needs a stable directory to find latest-epoch.ckpt.
RUN_ID="p512_geo_concat_auto_resume"

EXPERIMENT="pansplat-512-mix-spe-mona-multi-wsmse-geo-concat"

# Total target epochs for this 512 stage.
MAX_EPOCHS=10

# Initial checkpoint: your 256-resolution best checkpoint.
# First 512 fine-tuning run should use this through model.weights_path.
INITIAL_CKPT="/mnt/disk1/zwh/Project/PanSplat/logs/hk0l83e3/checkpoints/best-19-101439.ckpt"

# Auto-resume checkpoint generated during 512 training.
RUN_DIR="${PROJECT_DIR}/logs/${RUN_ID}"
CKPT_DIR="${RUN_DIR}/checkpoints"
LATEST_EPOCH_CKPT="${CKPT_DIR}/latest-epoch.ckpt"

# Restart control.
MAX_RESTARTS=20
SLEEP_SECONDS=10

# Training memory options.
MODEL_VIEW_BATCH=10
LOSS_VIEW_BATCH=10

# Log file for this auto-resume controller.
AUTO_LOG="${RUN_DIR}/auto_resume.log"
mkdir -p "${RUN_DIR}"

# ----------------------------
# Environment settings
# ----------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Strongly recommended for unstable proxy/network.
# You can sync W&B after training using:
#   wandb sync logs/${RUN_ID}/wandb
export WANDB_MODE=offline

# If old proxy variables are present, they may still affect W&B / requests.
# In offline mode they usually do not matter, but clearing them avoids surprises.
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
unset http_proxy
unset https_proxy
unset all_proxy

# ----------------------------
# Stop cleanly on Ctrl+C
# ----------------------------
stop_requested=0

handle_interrupt() {
  echo ""
  echo "[auto-resume] Interrupt received. Stop auto-resume loop."
  stop_requested=1
}

trap handle_interrupt INT TERM

restart_count=0

while true; do
  echo "============================================================" | tee -a "${AUTO_LOG}"
  echo "[auto-resume] Start time: $(date)" | tee -a "${AUTO_LOG}"
  echo "[auto-resume] Restart count: ${restart_count}/${MAX_RESTARTS}" | tee -a "${AUTO_LOG}"
  echo "[auto-resume] Project dir: ${PROJECT_DIR}" | tee -a "${AUTO_LOG}"
  echo "[auto-resume] Run ID: ${RUN_ID}" | tee -a "${AUTO_LOG}"
  echo "[auto-resume] GPU ID: ${GPU_ID}" | tee -a "${AUTO_LOG}"
  echo "============================================================" | tee -a "${AUTO_LOG}"

  if [ "${stop_requested}" -eq 1 ]; then
    echo "[auto-resume] Stop requested before launch." | tee -a "${AUTO_LOG}"
    exit 130
  fi

  if [ -f "${LATEST_EPOCH_CKPT}" ]; then
    echo "[auto-resume] Found latest epoch checkpoint." | tee -a "${AUTO_LOG}"
    echo "[auto-resume] Resume full training state from:" | tee -a "${AUTO_LOG}"
    echo "[auto-resume] ${LATEST_EPOCH_CKPT}" | tee -a "${AUTO_LOG}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    WANDB_RUN_ID=${RUN_ID} \
    python -m src.main \
      +experiment=${EXPERIMENT} \
      checkpointing.load=${LATEST_EPOCH_CKPT} \
      model.encoder.geometry_reliability.return_debug=false \
      model.decoder.view_batch=${MODEL_VIEW_BATCH} \
      loss.pyimage.decoder.view_batch=${LOSS_VIEW_BATCH} \
      trainer.max_epochs=${MAX_EPOCHS} \
      checkpointing.save_latest_epoch=true \
      2>&1 | tee -a "${AUTO_LOG}"

    exit_code=${PIPESTATUS[0]}

  else
    echo "[auto-resume] latest-epoch.ckpt not found." | tee -a "${AUTO_LOG}"
    echo "[auto-resume] Start 512 fine-tuning from 256 best weights:" | tee -a "${AUTO_LOG}"
    echo "[auto-resume] ${INITIAL_CKPT}" | tee -a "${AUTO_LOG}"

    if [ ! -f "${INITIAL_CKPT}" ]; then
      echo "[auto-resume] ERROR: INITIAL_CKPT not found:" | tee -a "${AUTO_LOG}"
      echo "[auto-resume] ${INITIAL_CKPT}" | tee -a "${AUTO_LOG}"
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
      trainer.max_epochs=${MAX_EPOCHS} \
      checkpointing.save_latest_epoch=true \
      2>&1 | tee -a "${AUTO_LOG}"

    exit_code=${PIPESTATUS[0]}
  fi

  echo "============================================================" | tee -a "${AUTO_LOG}"
  echo "[auto-resume] Process exited at: $(date)" | tee -a "${AUTO_LOG}"
  echo "[auto-resume] Exit code: ${exit_code}" | tee -a "${AUTO_LOG}"
  echo "============================================================" | tee -a "${AUTO_LOG}"

  if [ "${stop_requested}" -eq 1 ]; then
    echo "[auto-resume] Stop requested. Exit." | tee -a "${AUTO_LOG}"
    exit 130
  fi

  if [ "${exit_code}" -eq 0 ]; then
    echo "[auto-resume] Training finished normally. Stop." | tee -a "${AUTO_LOG}"
    exit 0
  fi

  if [ "${exit_code}" -eq 130 ] || [ "${exit_code}" -eq 143 ]; then
    echo "[auto-resume] Training interrupted manually. Stop." | tee -a "${AUTO_LOG}"
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