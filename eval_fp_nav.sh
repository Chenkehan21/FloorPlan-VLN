#!/bin/bash

CHUNKS=8
MODEL_PATH="FloorPlan-VLN/models/fp-nav-vision-r2r-rxr-lr-4e-5-videoframe6" # specify the path of your finetuned model here


#R2R
CONFIG_PATH="VLN_CE/vlnce_baselines/config/r2r_baselines/fp-nav.yaml"
SAVE_PATH="tmp/fp-nav-evaluation"


for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo $(( IDX % $CHUNKS ))
    CUDA_VISIBLE_DEVICES=$(( IDX % $CHUNKS )) python run.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --split-id $IDX \
    --model-path $MODEL_PATH \
    --result-path $SAVE_PATH &
    
done

wait