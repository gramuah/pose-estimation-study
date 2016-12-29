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
NET=$2
NET_lc=${NET,,}
ITERS=200000
DATASET_TRAIN=3Dplus_trainval
DATASET_TEST=3Dplus_test

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/PASCAL3D_CONTINUOUS_FC7.`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

NET_INIT=data/pascal_models/${NET}/train/vgg16_faster_rcnn_iter_70000.caffemodel
#NET_INIT=data/pascal_models/${NET}/trainval/vgg16_faster_rcnn_iter_70000.caffemodel
#NET_INIT=data/imagenet_models/${NET}.v2.caffemodel

time ./tools/train_net.py \
	--gpu ${GPU_ID} \
	--solver models/VGG16/faster_rcnn_end2end/continuous_fc7_solver_3Dplus.prototxt \
	--weights ${NET_INIT} \
	--imdb ${DATASET_TRAIN} \
	--iters ${ITERS} \
	--cfg experiments/cfgs/faster_rcnn_end2end.yml

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/VGG16/faster_rcnn_end2end/continuous_fc7_test_3Dplus.prototxt \
  --net ${NET_FINAL} \
  --imdb ${DATASET_TEST} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
