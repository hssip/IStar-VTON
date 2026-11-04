#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0
ARRY=(${CUDA_VISIBLE_DEVICES//,/ })
NUM_GPUS=${#ARRY[@]}
TORCH_DISTRIBUTED_DEBUG=INFO
MASTER_PORT=19895
PROJECT_ROOT=./

TRAIN_NAME=v1
DATA_NAME=vton_dresscode #"vton_dresscode","dresscode","vton"
LOG_NAME=train_$(date +%Y_%m_%d_%H_%M_%S)_${TRAIN_NAME}_${DATA_NAME}
LOG_FILENAME=${LOG_NAME}.out

nohup torchrun \
    --master_port $MASTER_PORT \
    --nproc_per_node $NUM_GPUS \
    --nnodes 1 \
    --node_rank 0 \
    image_train.py \
    --batch_size 2 \
    --accumulate 3 \
    --epochs 45 \
    --save_interval 100000 \
    --data_dir ${PROJECT_ROOT}/data \
    --work_dir ${PROJECT_ROOT}/output/${LOG_NAME} \
    --lr 2e-4 \
    --min_lr_rate 0.1 \
    --n_gpus $NUM_GPUS \
    --image_size 512 \
    --vit_image_size 224 \
    --vit_size vitg14 \
    --decode_layer 4 \
    --model_type IStar_XL \
    --diffusion_steps 1000 \
    --opt_type adan \
    --vae_model_path flux.2-dev \
    --image_encoder_path torch_dinov2_large/checkpoints \
    --scene_encoder_path bert-base-uncased \
    --max_scene_length 128 \
    --log_interval 30 \
    --dataset_kind $DATA_NAME \
    --cat_type edge \
    --crop_cloth True \
    --seed 3407 \
    --weight_decay 0.0 \
    --warm_up_ratio 0.001 \
    --mask_bkg False \
    --lr_sche_type cosine \
    --mask_ratio 0.2 \
    2>&1 | tee -a ./log/train/$LOG_FILENAME &

