#!/bin/bash
# 用法示例：
# ./resume.sh 6,7 \
#   /data/huangguang/checkpoint/vla-adapter/11/sandwich--265000_chkpt \
#   /data/huangguang/checkpoint/vla-adapter/12/sandwich \
#   /data/share_nips/robot/aidlux/cxg/sandwich_1111/data.json,/data/share_nips/robot/aidlux/cxg/sandwich_1112/data.json

set -e

# === 参数解析 ===
gpuid=$1         # 第一个参数：GPU ID 列表，例如 "0,1" 或 "6,7"
checkpoint=$2    # 第二个参数：checkpoint路径
save=$3          # 第三个参数：保存路径前缀
json_path=$4     # 第四个参数：数据 json 路径（逗号分隔）

if [ -z "$gpuid" ] || [ -z "$checkpoint" ] || [ -z "$save" ] || [ -z "$json_path" ]; then
    echo "Usage: $0 <gpu_ids> <checkpoint_path> <save_prefix> <json_paths>"
    echo "Example: $0 6,7 /path/to/ckpt /path/to/save /path/to/data1.json,/path/to/data2.json"
    exit 1
fi

# === 提取 resume_step ===
if [[ $(basename "$checkpoint") =~ ([0-9]+)_chkpt ]]; then
    resume_step=${BASH_REMATCH[1]}
else
    resume_step=0
fi

# === 拆解 save 参数 ===
run_root_dir=$(dirname "$save")"/"
run_id_note=$(basename "$save")

cd /data/huangguang/VLA-Adapter

# === 启动训练 ===
CUDA_VISIBLE_DEVICES=$gpuid \
torchrun --standalone --nnodes 1 --nproc-per-node $(echo $gpuid | awk -F',' '{print NF}') vla-scripts/finetune.py \
    --data_root_dir /data/huangguang/data/openvla/modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_lora True \
    --use_fz False \
    --image_aug True \
    --num_steps_before_decay 200000 \
    --max_steps 200005 \
    --save_freq 5000 \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training True \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lora_rank 64 \
    --use_pro_version True \
	--patch True \
    --use_minivlm False \
    --resume True \
    --config_file_path "$checkpoint" \
    --resum_vla_path "$checkpoint" \
    --vlm_path "$checkpoint" \
    --resume_step "$resume_step" \
    --run_root_dir "$run_root_dir" \
    --run_id_note "$run_id_note" \
    --json_path "$json_path"