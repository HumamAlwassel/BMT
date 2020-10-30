#!/bin/bash --login

DEVICE_IDS=$1           
VIDEO_FEATURES_PATH=$2

VIDEO_FEATURE_NAME=$(basename $VIDEO_FEATURES_PATH .h5)
EXP_NAME=$(uuidgen)

echo "DEVICE_IDS ${DEVICE_IDS}"
echo "VIDEO_FEATURES_PATH ${VIDEO_FEATURES_PATH}"
echo "VIDEO_FEATURE_NAME ${VIDEO_FEATURE_NAME}"
echo "EXP_NAME ${EXP_NAME}"

D_VID=512
FEATURE_TIMESPAN_IN_FPS=16
FPS_AT_EXTRACTION=15
PAD_VIDEO_FEATS_UP_TO=720
EARLY_STOP_AFTER=10

####################################
###### STAGE 1: TRAIN CAPTION ######
####################################

BATCH_SIZE=128
LR=0.0002

python main.py \
    --procedure train_cap \
    --exp_name $EXP_NAME \
    --video_feature_name $VIDEO_FEATURE_NAME \
    --video_features_path $VIDEO_FEATURES_PATH \
    --d_vid $D_VID \
    --feature_timespan_in_fps $FEATURE_TIMESPAN_IN_FPS \
    --fps_at_extraction $FPS_AT_EXTRACTION \
    --pad_video_feats_up_to $PAD_VIDEO_FEATS_UP_TO \
    --B $BATCH_SIZE \
    --lr $LR \
    --early_stop_after $EARLY_STOP_AFTER \
    --device_ids $DEVICE_IDS

####################################
##### STAGE 2: TRAIN PROPOSALS #####
####################################

BATCH_SIZE=64
LR=0.0002
PRETRAINED_CAP_MODEL_PATH=./log/${VIDEO_FEATURE_NAME}/train_cap/${EXP_NAME}/best_cap_model.pt

python main.py \
    --procedure train_prop \
    --exp_name $EXP_NAME \
    --video_feature_name $VIDEO_FEATURE_NAME \
    --video_features_path $VIDEO_FEATURES_PATH \
    --d_vid $D_VID \
    --feature_timespan_in_fps $FEATURE_TIMESPAN_IN_FPS \
    --fps_at_extraction $FPS_AT_EXTRACTION \
    --pad_video_feats_up_to $PAD_VIDEO_FEATS_UP_TO \
    --B $BATCH_SIZE \
    --lr $LR \
    --early_stop_after $EARLY_STOP_AFTER \
    --pretrained_cap_model_path $PRETRAINED_CAP_MODEL_PATH \
    --device_ids $DEVICE_IDS
