#!/bin/bash

# 1. 외부 인자 받기
model_name="$1"  # ex) df_circ_ss or df_circ_ss_10
device="$2"      # ex) mlm, mlm_ntp, etc.
target_model="$3"          # ex) True or False
name="$4"  # ex) experiment name
batch_size="$5"  # ex) 32 or 64

if [ -z "$name" ]; then
    name=""
fi

if [ -z "$target_model" ]; then
    target_model="rnabert"
fi

# 3. 고정 파라미터 설정
d_model=64
if [ -z "$batch_size" ]; then
    batch_size=64
fi
seeds=(1 2 3)
task="both"
echo "Running experiments with model: $model_name on device $device"

# 4. 실험 루프
for seed in "${seeds[@]}"; do
    echo "======================================"
    echo "Running $model_name with seed $seed"
    echo "======================================"

    echo "[Task: $task]"
    # exp_name="${model_name}_s${seed}_${task}"
    exp_name="${task}_${name}_${target_model}"
    echo "Experiment Name: $exp_name"

    python training.py \
        --model_name "$model_name" \
        --target_model "$target_model" \
        --device "$device" \
        --seed "$seed" \
        --task "$task" \
        --exp "$exp_name" \
        --is_cross_attention \
        --verbose \
        --d_model "$d_model" \
        --batch_size "$batch_size"

    echo "--------------------------------------"
    echo "Completed all tasks for seed $seed"
    echo "--------------------------------------"
done
