#!/bin/bash --login

DEVICE_IDS=$1           
VIDEO_FEATURES_PATH=$2
PRETRAINED_CAP_MODEL_PATH=$3
PROP_PRED_PATH=$4

VIDEO_FEATURE_NAME=$(basename $VIDEO_FEATURES_PATH .h5)
EXP_NAME=$(uuidgen)

echo "DEVICE_IDS ${DEVICE_IDS}"
echo "VIDEO_FEATURES_PATH ${VIDEO_FEATURES_PATH}"
echo "VIDEO_FEATURE_NAME ${VIDEO_FEATURE_NAME}"
echo "PRETRAINED_CAP_MODEL_PATH ${PRETRAINED_CAP_MODEL_PATH}"
echo "PROP_PRED_PATH ${PROP_PRED_PATH}"
echo "EXP_NAME ${EXP_NAME}"

D_VID=512
FEATURE_TIMESPAN_IN_FPS=16
FPS_AT_EXTRACTION=15
PAD_VIDEO_FEATS_UP_TO=720
EARLY_STOP_AFTER=10

# print the results using GT proposals
python get_metrics_using_gt_proposals.py \
    --checkpoint $PRETRAINED_CAP_MODEL_PATH

# compute the results using learned proposals
python main.py \
    --procedure evaluate \
    --prop_pred_path $PROP_PRED_PATH \
    --exp_name $EXP_NAME \
    --video_feature_name $VIDEO_FEATURE_NAME \
    --video_features_path $VIDEO_FEATURES_PATH \
    --d_vid $D_VID \
    --feature_timespan_in_fps $FEATURE_TIMESPAN_IN_FPS \
    --fps_at_extraction $FPS_AT_EXTRACTION \
    --pad_video_feats_up_to $PAD_VIDEO_FEATS_UP_TO \
    --early_stop_after $EARLY_STOP_AFTER \
    --pretrained_cap_model_path $PRETRAINED_CAP_MODEL_PATH \
    --device_ids $DEVICE_IDS
