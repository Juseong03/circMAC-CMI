#!/bin/bash
# 사용 예: ./run_pretrain_task.sh df_circ_ss mlm

# 1. 외부 인자 받기
data_file="$1"  # ex) df_circ_ss or df_circ_ss_10
model_name="$2"
task_key="$3"   # ex) mlm, mlm_ntp, etc.
device="$4"      # ex) -1, 0, 1 etc.
name="$5"  # ex) experiment name



# 2. 입력 검증
if [ -z "$data_file" ] || [ -z "$task_key" ]; then
    echo "❌ 사용법: ./run_pretrain_task.sh <data_file> <task_key>"
    echo "예시: ./run_pretrain_task.sh df_circ_ss mlm_ntp"
    exit 1
fi

# 3. data_file 유효성 체크
if [[ "$data_file" != "df_circ_ss" && "$data_file" != "df_circ_ss_5" ]]; then
    echo "❌ 지원되지 않는 data_file: '$data_file'"
    echo "지원되는 값: df_circ_ss, df_circ_ss_5"
    exit 1
fi

if [[ "$model_name" != "circmac" && "$model_name" != "mamba" && "$model_name" != "transformer" && "$model_name" != "hymba" && "$model_name" != "lstm" ]]; then
    echo "지원되지 않는 model_name: '$model_name'"
    echo "지원되는 값: circmac, mamba, transformer, hymba, lstm"
    exit 1
fi

if [[ "$name" == "None" ]]; then
    name=""
fi

# 4. 기본 설정
# model_name="circmac"
seed=1
batch_size=64
max_len=1022
d_model=64
n_layer=4
kmer=1
ssp_vocab_size=4
epochs=300
earlystop=30
lr=1e-4


# 5. task flag 정의
declare -A tasks
tasks["mlm"]="--mlm"
tasks["ntp"]="--ntp"
tasks["ssp"]="--ssp"
tasks["ssl"]="--ss_labels"
tasks["sslm"]="--ss_labels_multi"


tasks["mlm_ntp"]="--mlm --ntp"
tasks["mlm_ssp"]="--mlm --ssp"
tasks["mlm_ssl"]="--mlm --ss_labels"
tasks["mlm_ssp_ssl"]="--mlm --ssp --ss_labels"
tasks["mlm_ntp_ssp"]="--mlm --ntp --ssp"
tasks["mlm_ntp_ssp_ssl"]="--mlm --ntp --ssp --ss_labels"
tasks["mlm_sslm"]="--mlm --ss_labels_multi"
tasks["mlm_ssp_sslm"]="--mlm --ssp --ss_labels_multi"
tasks["mlm_ssp_sslm_pair"]="--mlm --ssp --ss_labels_multi --pairing"


tasks["mlm_ntp_sslm_pair"]="--mlm --ntp --ss_labels_multi --pairing"
tasks["mlm_ntp_ssp_sslm_pair"]="--mlm --ntp --ssp --ss_labels_multi --pairing"
tasks["mlm_ntp_ssp_sslm"]="--mlm --ntp --ssp --ss_labels_multi"

# New tasks: CPCL and BSJ_MLM
tasks["cpcl"]="--cpcl"
tasks["bsj_mlm"]="--bsj_mlm"
tasks["mlm_cpcl"]="--mlm --cpcl"
tasks["mlm_bsj"]="--mlm --bsj_mlm"
tasks["mlm_cpcl_bsj"]="--mlm --cpcl --bsj_mlm"
tasks["mlm_cpcl_pair"]="--mlm --cpcl --pairing"
tasks["mlm_bsj_pair"]="--mlm --bsj_mlm --pairing"
tasks["mlm_cpcl_bsj_pair"]="--mlm --cpcl --bsj_mlm --pairing"
tasks["full"]="--mlm --ntp --ssp --ss_labels_multi --pairing --cpcl --bsj_mlm"


if [[ -z "${tasks[$task_key]}" ]]; then
    echo "❌ Unknown task key: '$task_key'"
    echo "지원되는 task keys: ${!tasks[@]}"
    exit 1
fi

# 6. 실행
exp_name="${name}_${task_key}_pretrained"
echo "▶ Running: $exp_name"

python pretraining.py \
    --model_name "$model_name" \
    --device "$device" \
    --seed "$seed" \
    --exp "$exp_name" \
    --data_file "$data_file" \
    --max_len "$max_len" \
    --kmer "$kmer" \
    --batch_size "$batch_size" \
    --lr "$lr" \
    --epochs "$epochs" \
    --earlystop "$earlystop" \
    --d_model "$d_model" \
    --n_layer "$n_layer" \
    --ssp_vocab_size "$ssp_vocab_size" \
    --verbose \
    ${tasks[$task_key]}

echo "✅ Done: $exp_name"
