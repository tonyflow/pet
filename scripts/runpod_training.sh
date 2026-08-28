#!/usr/bin/env bash

set -Eeuo pipefail

readonly workspace="${PET_WORKSPACE:-/workspace}"
readonly output_root="${PET_OUTPUT_ROOT:-$workspace/artifacts/training}"
readonly data_config="${PET_DATA_CONFIG:-/app/configs/data/oxford_pet_runpod.yaml}"
readonly classification_config="${PET_CLASSIFICATION_CONFIG:-/app/configs/training/classification_runpod.yaml}"
readonly segmentation_config="${PET_SEGMENTATION_CONFIG:-/app/configs/training/segmentation_runpod.yaml}"
readonly classification_run_name="${PET_CLASSIFICATION_RUN_NAME:?Set PET_CLASSIFICATION_RUN_NAME}"
readonly segmentation_run_name="${PET_SEGMENTATION_RUN_NAME:?Set PET_SEGMENTATION_RUN_NAME}"
readonly classification_run_dir="$output_root/$classification_run_name"
readonly segmentation_run_dir="$output_root/$segmentation_run_name"

mkdir -p "$workspace/logs" "$output_root"
exec > >(tee -a "$workspace/logs/training.log") 2>&1

on_error() {
  local status="$?"
  printf 'TRAINING_FAILED exit_code=%s\n' "$status"
  printf '%s\n' "$status" > "$workspace/TRAINING_FAILED"
  printf 'The container will remain alive for inspection until the Pod is deleted or capped.\n'
  sleep infinity
}
trap on_error ERR

rm -f "$workspace/TRAINING_COMPLETE" "$workspace/TRAINING_FAILED"
printf 'TRAINING_STARTED\n'
python -m pet.cli.prepare_data --config "$data_config"

classification_resume=()
if [[ -f "$classification_run_dir/checkpoints/latest.pt" ]]; then
  classification_resume=(--resume "$classification_run_dir/checkpoints/latest.pt")
fi
PET_RUN_GIT_REVISION="${PET_CLASSIFICATION_GIT_REVISION:-${PET_GIT_REVISION:-}}" \
  python -m pet.cli.train \
    --training-config "$classification_config" \
    --data-config "$data_config" \
    --output-root "$output_root" \
    --run-name "$classification_run_name" \
    --device cuda \
    "${classification_resume[@]}"

segmentation_resume=()
if [[ -f "$segmentation_run_dir/checkpoints/latest.pt" ]]; then
  segmentation_resume=(--resume "$segmentation_run_dir/checkpoints/latest.pt")
fi
python -m pet.cli.train \
  --training-config "$segmentation_config" \
  --data-config "$data_config" \
  --output-root "$output_root" \
  --run-name "$segmentation_run_name" \
  --device cuda \
  "${segmentation_resume[@]}"

touch "$workspace/TRAINING_COMPLETE"
printf 'TRAINING_COMPLETE\n'
sleep infinity
