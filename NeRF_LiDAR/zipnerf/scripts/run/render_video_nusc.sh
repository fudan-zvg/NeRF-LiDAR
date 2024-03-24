#!/bin/bash

CONFIG=$4
IFS="." read -ra parts <<< "$CONFIG"
prefix="${parts[0]}"

SCENE=$3
EXPERIMENT=$prefix/"$SCENE"/"$2"

DATA_ROOT=$DATA_ROOT
DATA_DIR="$DATA_ROOT"/"$SCENE"
CONFIG=$4
IFS="." read -ra parts <<< "$CONFIG"
prefix="${parts[0]}"
let render_chunk_size=16384*$1

address=$(shuf -i 10000-60000 -n 1)

if [ $1 -eq 1 ];then
echo 'one gpu rendering'
# accelerate launch --num_processes $1  --main_process_port 20245 render.py --gin_configs=configs/nuscenes.gin \
python render_video.py --gin_configs=configs/${CONFIG} \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.sample_n_test = 7" \
  --gin_bindings="Config.sample_m_test = 3" \
  --gin_bindings="Config.render_path = False" \
  --gin_bindings="Config.render_path_frames = 12" \
  --gin_bindings="Config.render_video_fps = 6" \
  --gin_bindings="Config.factor = 1" \
  --gin_bindings="Config.render_chunk_size = $render_chunk_size"
else
echo 'distributed rendering'
accelerate launch --num_processes $1 --main_process_port $address render_video.py --gin_configs=configs/${CONFIG} \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.sample_n_test = 7" \
  --gin_bindings="Config.sample_m_test = 3" \
  --gin_bindings="Config.render_path = False" \
  --gin_bindings="Config.render_path_frames = 12" \
  --gin_bindings="Config.render_video_fps = 6" \
  --gin_bindings="Config.factor = 1" \
  --gin_bindings="Config.render_chunk_size = $render_chunk_size"
fi