#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen-2.5-VL-7B-Instruct # Using HuggingFace model ID or download the model and put it in the correct path as mentioned in README.

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration
# datasets=floorplan_r2r
datasets=floorplan_r2r,floorplan_rxr

# Output configuration
run_name="floorplan-vision-r2r-rxr-lr-4e-5-videoframe6-finetune1"
output_dir=./output-floorplan-vision-r2r-rxr-lr-4e-5-videoframe6-finetune1

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --learning_rate 4e-5     \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --base_interval 1 \
    --video_max_frames 6 \
    --video_min_frames 1 \
    --max_pixels 200704 \
    --min_pixels 784 \
    --video_max_frame_pixels 352800
    --video_min_frame_pixels 200704
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name}"

# Launch training
NPROC_PER_NODE=4
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}