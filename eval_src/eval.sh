export CUDA_VISIBLE_DEVICES=1
EVAL_NAME=eval_1
EVAL_DATA=vitonhd              #vitonhd dresscode
EVAL_DATA_CATE=upper_body              #vitonhd -> mix|||dresscode -> dresses/lower_body/upper_body
TRAIN_NAME=train_1 
PAIR_LABEL=paired                 #unpaired
GEN_NAME=gen_1
VERSION_NUM=train_1



LOG_NAME=eval_$(date +%Y_%m_%d_%H_%M_%S).out
python -u eval.py \
    --gen_folder ./ \
    --gt_folder ./data/zalando-hd-resized/test/image \
    --dresscode_dataroot ./data/DressCode \
    --vitonhd_dataroot ./data/zalando-hd-resized \
    --streetvton ./data/streetvton \
    --test_order paired \
    --dataset ${EVAL_DATA} \
    --category upper_body  \
    --batch_size 8 \
    --workers 8 \
    --eval_image_height 1024 \
    --eval_image_width 768 \
    2>&1 | tee -a ./log/$LOG_NAME