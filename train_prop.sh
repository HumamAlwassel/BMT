#!/bin/bash --login

#DEVICE_IDS=0
#BATCH_SIZE=16
#LR=0.00005

DEVICE_IDS=0
BATCH_SIZE=128
LR=0.0001
VIDEO_FEATURE_NAME=i3d
PRETRAINED_CAP_MODEL_PATH=./log/train_cap/1029203419/best_cap_model.pt

python main.py \
    --procedure train_prop \
    --video_feature_name $VIDEO_FEATURE_NAME \
    --pretrained_cap_model_path $PRETRAINED_CAP_MODEL_PATH \
    --B $BATCH_SIZE \
    --lr $LR \
    --early_stop_after 10 \
    --device_ids $DEVICE_IDS
