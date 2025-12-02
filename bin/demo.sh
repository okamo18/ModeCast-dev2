#!/bin/sh

model=modecast
input_dir=Robot/robot_csv
fn=recording_2021_12_15_20H_29M.csv
root_out_dir=out/

lstep=13
lcurr=60
lrprt=3
h=10 #hが大きいほど、データが複雑になるから、でかいと過学習　、小さいと学習できづらい
err_th=0.8 #レジームを切るしきい値
trunc_th=0.99 #シグマカットオフのしきい値
window=10 #使ってない
train_size=73  #  >= lcurr + lstep

COMMAND="poetry run python -m src.main --multirun \
  model=$model \
  mlflow.experiment_name=dmd_demo_${model} \
  io.fn=${fn} \
  io.input_dir=${input_dir} \
  model.h=${h} \
  model.err_th=${err_th} \
  model.trunc_th=${trunc_th} \
  model.lcurr=${lcurr} \
  model.lstep=${lstep} \
  model.lrprt=${lrprt} \
  model.train_size=${train_size} \
  model.window=${window} \
  dark_mode=False \
  viz=False \
  save=True \
  verbose=True"

bash bin/run_wrapper.sh "$@" "$COMMAND"