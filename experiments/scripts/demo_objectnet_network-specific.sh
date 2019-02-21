#!/bin/bash
# Usage:
# ./experiments/scripts/default_faster_rcnn.sh GPU NET [--set ...]
# Example:
# ./experiments/scripts/default_faster_rcnn.sh 0 ZF \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET_TEST=ObjectNet3D_test

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/OBJECTNET_NETWORK_SPECIFIC_(train-val).txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

NET_FINAL='data/demo_models/SPECIFIC-NETWORK_OBJECTNET.caffemodel'

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/VGG16/faster_rcnn_end2end/test_objectnet_network-specific.prototxt \
  --net ${NET_FINAL} \
  --imdb ${DATASET_TEST} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --continuous \
  ${EXTRA_ARGS}
