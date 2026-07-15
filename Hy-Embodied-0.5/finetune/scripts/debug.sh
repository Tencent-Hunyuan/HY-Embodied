#!/bin/bash
# Single-node, single-GPU training script for HunYuanVLMoT fine-tuning.
#
# Usage:
#   bash scripts/debug.sh
#
# By default this trains on the small packed test JSON in tests/, which
# exercises the data-packing path (cu_seqlens with multiple segments per
# batch). Swap `datasets` below to point at your own JSON for a real run.

echo "current Python: $(which python3)"
echo "current Python version: $(python3 --version)"

# Dynamically resolve project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"

# Distributed / NCCL settings
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore::UserWarning"

# ViT optimization toggles (see set_model() in train_qwen.py)
export VIT_WITH_GRAD=1
export UPDATE_VIT_EVERY_N=5

# Each rank draws its own random samples (use only with pre-sharded datasets)
export SAMPLE_INDEPENDENTLY=1

# ---- Dataset --------------------------------------------------------------
# Two ways to enable packing:
#   (A) Pre-pack in JSON: each entry is a list of dicts → _get_packed_item
#       concatenates them inside one __getitem__ call. Use batch_size=1.
#   (B) Online-pack via collator: each entry is a single dict and
#       per_device_train_batch_size > 1 → FlattenedDataCollator concatenates
#       them. Use batch_size=N.
datasets=${PROJECT_ROOT}/tests/test_data_packing.json
batch_size=1

# Alternative: online-packed test data
# datasets=${PROJECT_ROOT}/tests/test_data.json
# batch_size=128

# ---- Model & training config ----------------------------------------------
llm=tencent/HY-Embodied-0.5
deepspeed=${PROJECT_ROOT}/scripts/zero0.json

lr=5e-5
grad_accum_steps=1

run_name="debug_single_gpu_verify"
output_dir=/tmp/debug_train_output/$run_name

cd ${PROJECT_ROOT}/

echo ""
echo "============================================================"
echo "  Starting training"
echo "  Dataset: ${datasets}"
echo "  Model:   ${llm}"
echo "  Output:  ${output_dir}"
echo "============================================================"
echo ""

# Verify key files exist
echo "[CHECK] chat_template_train.jinja exists: $(test -f ${PROJECT_ROOT}/chat_template_train.jinja && echo YES || echo NO)"
echo "[CHECK] dataset exists:                   $(test -f ${datasets} && echo YES || echo NO)"
echo "[CHECK] model config exists:              $(test -f ${llm}/config.json && echo YES || echo NO)"
echo ""

torchrun --nproc_per_node=8 --master_port=29599 ${PROJECT_ROOT}/qwenvl/train/train_qwen.py \
    --model_name_or_path "${llm}" \
    --deepspeed ${deepspeed} \
    --dataset_use ${datasets} \
    --group_by_modality_length False \
    --data_flatten True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --vision_tower_lr 5e-6 \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 4194304 \
    --min_pixels 25088 \
    --video_max_frames 32 \
    --video_min_frames 8 \
    --video_max_pixels 134217728 \
    --video_min_pixels 100352 \
    --video_fps 2.0 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --min_lr_ratio 1.0 \
    --weight_decay 0.0001 \
    --warmup_ratio 0.0 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --run_name ${run_name} \
    --report_to none
