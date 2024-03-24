#!/bin/bash
address=$(shuf -i 10000-60000 -n 1)

CONFIG=$4
IFS="." read -ra parts <<< "$CONFIG"
prefix="${parts[0]}"

SCENE=$3
EXPERIMENT=$prefix/"$SCENE"/"$2"
DATA_ROOT=$DATA_ROOT
DATA_DIR="$DATA_ROOT"/"$SCENE"

# BATCH_SIZE=8192
BATCH_SIZE=16384
RENDER_CHUNK_SIZE=4096
max_steps=40000

let BATCH_SIZE=$BATCH_SIZE
let RENDER_CHUNK_SIZE=$RENDER_CHUNK_SIZE*$1

echo 'CONFIG' ${CONFIG}
echo 'EXPERIMENT' ${EXPERIMENT}

if [ $1 -eq 1 ];then
echo 'one gpu training (for debug)'
BATCH_SIZE=4096
python train.py --gin_configs=configs/${CONFIG} \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.batch_size = ${BATCH_SIZE}" \
  --gin_bindings="Config.factor = 1" \
  --gin_bindings="Config.render_chunk_size = ${RENDER_CHUNK_SIZE}" \
  --gin_bindings="Config.max_steps = $max_steps" \
  --gin_bindings="Config.gpu_num = ${1}"
else
echo 'distributed training'
echo 'GPU num' ${1}
accelerate launch --num_processes $1 --main_process_port $address train.py --gin_configs=configs/${CONFIG} \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.batch_size = ${BATCH_SIZE}" \
  --gin_bindings="Config.factor = 1" \
  --gin_bindings="Config.render_chunk_size = ${RENDER_CHUNK_SIZE}" \
  --gin_bindings="Config.max_steps = $max_steps" \
  --gin_bindings="Config.gpu_num = ${1}"
fi

#  --gin_bindings="Config.lr_final = ${LR_FINAL}"