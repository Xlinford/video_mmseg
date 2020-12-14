#!/usr/bin/env bash

set -x

#CONFIG_FILE="configs/video/cityscapes/memory_r50-d16_769x769_80k_cityscapes_video.py"
#CONFIG_PY="${CONFIG_FILE##*/}"
#CONFIG="${CONFIG_PY%.*}"
#WORK_DIR="./work_dirs/${CONFIG}_k64_s4"
#WORK_DIR="./work_dirs/test"
WORK_DIR="./work_dirs/memory_r50-d8_769x769_80k_cityscapes_video"
CONFIG_FILE="${WORK_DIR}/memory_r50-d8_769x769_80k_cityscapes_video.py"
SHOW_DIR="${WORK_DIR}/show"
TMPDIR="${WORK_DIR}/tmp"
CHECKPOINT="${WORK_DIR}/latest.pth"
RESULT_FILE="${WORK_DIR}/result.pkl"
#CHECKPOINT="${WORK_DIR}/iter_36000.pth,${WORK_DIR}/iter_40000.pth,${WORK_DIR}/iter_32000.pth"
GPUS=1
PORT=${PORT:-29511}

if [ ! -d "${WORK_DIR}" ]; then
  mkdir -p "${WORK_DIR}"
  cp "${CONFIG_FILE}" $0 "${WORK_DIR}"
fi

echo -e "\nconfig file: ${CONFIG}\n"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

RANDOM_SEED=0
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=2
#export CUDA_VISIBLE_DEVICES=4,5,6,7

# training
#echo -e '\nDistributed Training\n'
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    ./tools/train.py ${CONFIG_FILE} \
#    --seed $RANDOM_SEED \
#    --launcher 'pytorch' \
#    --work-dir $WORK_DIR \
#    --resume-from $CHECKPOINT \
#echo -e "\nWork Dir: ${WORK_DIR}.\n"

# evaluation
#echo -e "\nEvaluating ${WORK_DIR}\n"
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#  ./tools/test.py \
#  ${CONFIG_FILE} \
#  ${CHECKPOINT} \
#  --launcher 'pytorch' \
#  --work-dir $WORK_DIR \
#  --eval mIoU  \
#  --tmpdir $TMPDIR \
#  --out $RESULT_FILE \
echo -e "\nWork Dir: ${WORK_DIR}.\n"

python ./tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --test-fps \
    --test-shape 769 769 \

# visualization
#echo -e '\nVisualization.\n'
#if [ -d "${SHOW_DIR}" ]; then
#  rm -rf "${SHOW_DIR}"
#  mkdir "${SHOW_DIR}"
#fi
#python ./tools/test.py \
#    ${CONFIG_FILE} \
#    ${CHECKPOINT} \
#    --show-dir $SHOW_DIR \
