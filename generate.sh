#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
TRAIN_DATA_NAME=vton              #vton/dresscode
TRAIN_NAME=train_1    
PAIR_LABEL=unpair         #unpair pair
GENERATED_DATASET=vitonhd #vitonhd \ dresscode
GEN_NAME=_gen1            
GENERATED_SUBDATA=dresses
MODEL_NAME=ema_XXX.pt

# process
UNPAIR=True
if [ "$PAIR_LABEL" = "pair" ]; then 
    UNPAIR=False
else
    UNPAIR=True
fi

LOG_NAME=generate_${GENERATED_DATASET}_${GENERATED_SUBDATA}_$(date +%Y_%m_%d_%H_%M_%S)_${TRAIN_NAME}_${TRAIN_DATA_NAME}_${PAIR_LABEL}${GEN_NAME}.out
nohup python generate.py \
    --data_dir ./data \
    --output_dir ./generation \
    --model_path ./output/${TRAIN_NAME}_${TRAIN_DATA_NAME}/${MODEL_NAME} \
    --vae_path stabilityai/sd-vae-ft-mse \
    --cloth_encoder_path torch_dinov2_large/checkpoints \
    --scene_encoder_path bert-base-uncased \
    --image_size 512 \
    --vit_img_size 224 \
    --vit_size vitg14 \
    --num_sampling_steps 30 \
    --cfg_scale 2.0 \
    --pow_scale 1.0 \
    --batch_size 30 \
    --unpair $UNPAIR \
    --max_scene_length 128 \
    --generate_dataset ${GENERATED_DATASET} \
    --generate_subdata ${GENERATED_SUBDATA} \
    --cat_type plain \
    --crop_cloth True \
    --gen_name $GEN_NAME \
    --model_type IStar_XL \
    2>&1 | tee -a ./log/generate/$LOG_NAME &