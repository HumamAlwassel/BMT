#!/bin/bash --login

#DEVICE_IDS=0
#BATCH_SIZE=32
#LR=0.00005

DEVICE_IDS="1 0"
BATCH_SIZE=128
LR=0.0002

python main.py \
    --procedure train_cap \
    --B $BATCH_SIZE \
    --lr $LR \
    --device_ids $DEVICE_IDS

