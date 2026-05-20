#!/usr/bin/env bash
set -e

cd /mnt/disk1/zwh/Project/PanSplat

GPU_ID=7
DATASET_ROOT=/mnt/disk1/zwh/Dataset/mix

SCENES=(
  block_fix
  360vo_seq3
  panocity_jinan_jinan_block27
  panocity_ningbo_ningbo_block35
)

MODELS=(
  "A|ablate-512-A|/mnt/disk1/zwh/Project/PanSplat/logs/ablate512_A/checkpoints/best-00-001600.ckpt"
  "B|ablate-512-B|/mnt/disk1/zwh/Project/PanSplat/logs/ablate512_B/checkpoints/best-00-001600.ckpt"
  "C|ablate-512-C|/mnt/disk1/zwh/Project/PanSplat/logs/ablate512_C/checkpoints/best-00-001600.ckpt"
  "A_B|ablate-512-A-B|/mnt/disk1/zwh/Project/PanSplat/logs/ablate512_A_B/checkpoints/best-00-001600.ckpt"
)

for MODEL_INFO in "${MODELS[@]}"; do
  IFS="|" read -r MODEL_NAME EXPERIMENT CKPT <<< "${MODEL_INFO}"

  if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    exit 1
  fi

  for SCENE in "${SCENES[@]}"; do
    RUN_ID="test_ablate512_${MODEL_NAME}_${SCENE}_ctx20_30"

    echo "============================================================"
    echo "[TEST] model=${MODEL_NAME}"
    echo "[TEST] experiment=${EXPERIMENT}"
    echo "[TEST] scene=${SCENE}"
    echo "[TEST] ckpt=${CKPT}"
    echo "[TEST] run_id=${RUN_ID}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    WANDB_RUN_ID=${RUN_ID} \
    python -m src.main \
      +experiment=${EXPERIMENT} \
      mode=test \
      checkpointing.load=${CKPT} \
      dataset=mix \
      dataset.roots=[${DATASET_ROOT}] \
      'dataset.train_scenes=[]' \
      'dataset.val_scenes=[]' \
      "dataset.test_scenes=[${SCENE}]" \
      dataset.view_sampler.min_distance_between_context_views=10 \
      dataset.view_sampler.max_distance_between_context_views=10 \
      dataset.view_sampler.test_times_per_scene=1 \
      "+dataset.view_sampler.chosen={${SCENE}:[20]}" \
      test.compute_scores=true \
      test.save_image=true \
      wandb.mode=offline
  done
done
