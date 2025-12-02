#!/bin/sh

#  ----------------------------------------  #
# | model0          : synthetic data       | #
# | model1 - model4 : motion capture data  | #
# | model99         : other (need to name) | #
#  ----------------------------------------  #

# set root path in .pth file
CURRENT_DIR="$(dirname "$0")"
CURRENT_DIR=$(realpath "$CURRENT_DIR/..")
PTH_FILE_PATH="${CURRENT_DIR}/.venv/lib/python3.9/site-packages/streamdmd.pth"
echo "${CURRENT_DIR}" > "$PTH_FILE_PATH"

# I/O processing
input_dir="${1:-ett}"
uuid="${2:-5}"
model=modecast

# experiment parameters
# 1. lstep=30, train_size=100
# 2. lstep=100, train_size=300
lstep="${6:-13}"
lcurr=60
lrprt=3
h="${3:-15}"
err_th="${4:-0.8}"
trunc_th="${5:-0.99}"
train_size=73  #  >= lcurr + lstep

python src/main.py \
    mlflow.experiment_name=dmd_demo_${model} \
    model=$model \
    io.input_dir=${input_dir} \
    io.uuid=${uuid} \
    model.h=${h} \
    model.err_th=${err_th} \
    model.trunc_th=${trunc_th} \
    model.lcurr=${lcurr} \
    model.lstep=${lstep} \
    model.lrprt=${lrprt} \
    model.train_size=${train_size} \
    dark_mode=True \
    viz=True \
    save=True \
    verbose=False \
