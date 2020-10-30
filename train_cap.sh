#!/bin/bash --login

#DEVICE_IDS=0
#BATCH_SIZE=32
#LR=0.00005

DEVICE_IDS=0
BATCH_SIZE=128
LR=0.0002
VIDEO_FEATURE_NAME=i3d

python main.py \
    --procedure train_cap \
    --video_feature_name $VIDEO_FEATURE_NAME \
    --B $BATCH_SIZE \
    --lr $LR \
    --early_stop_after 10 \
    --device_ids $DEVICE_IDS
