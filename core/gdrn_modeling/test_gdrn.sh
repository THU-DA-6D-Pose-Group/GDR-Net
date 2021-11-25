#!/usr/bin/env bash
# test
set -x

# commonly used opts:
# MODEL.WEIGHTS: resume or pretrained, or test checkpoint
CFG=$1
CUDA_VISIBLE_DEVICES=$2
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
# GPUS=($(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n'))
NGPU=${#GPUS[@]}  # echo "${GPUS[0]}"
echo "use gpu ids: $CUDA_VISIBLE_DEVICES num gpus: $NGPU"
CKPT=$3
if [ ! -f "$CKPT" ]; then
    echo "$CKPT does not exist."
    exit 1
fi
NCCL_DEBUG=INFO
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=$2 python core/gdrn_modeling/main_gdrn.py \
    --config-file $CFG --num-gpus $NGPU --eval-only \
    --opts MODEL.WEIGHTS=$CKPT \
    ${@:4}

# tensorboard --logdir /path/to/logdir --bind_all # --port 6007
# to see tensorboard logs locally:
# ssh -L 6006:localhost:6006 user@server
