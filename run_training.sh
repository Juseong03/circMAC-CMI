#!/bin/bash

# 1. 외부 인자 받기
model_name="$1"  # ex) df_circ_ss or df_circ_ss_10
device="$2"      # ex) mlm, mlm_ntp, etc.
rc="$3"          # ex) True or False

# 2. --is_rc 인자 조건 설정
if [ "$rc" == "True" ]; then
    is_rc="--rc"
    rc_name='rc'
else
    is_rc=""
    rc_name='no_rc'
fi

# 3. 고정 파라미터 설정
d_model=64
batch_size=64
seeds=(1 2 3)

echo "Running experiments with model: $model_name on device $device"

# 4. 실험 루프
for seed in "${seeds[@]}"; do
    echo "======================================"
    echo "Running $model_name with seed $seed"
    echo "======================================"

    for task in binding sites both; do
        echo "[Task: $task]"
        exp_name="${task}_${rc_name}"

        python training.py \
            --model_name "$model_name" \
            --device "$device" \
            --seed "$seed" \
            --task "$task" \
            --exp "$exp_name" \
            --is_cross_attention \
            --verbose \
            $is_rc \
            --d_model "$d_model" \
            --batch_size "$batch_size"
    done

    echo "--------------------------------------"
    echo "Completed all tasks for seed $seed"
    echo "--------------------------------------"
done
