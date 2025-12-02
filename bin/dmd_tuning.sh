#!/bin/sh

#  ---------------------------------------  #
# | model0          : synthetic data      | #
# | model1 - model4 : motion capture data | #
#  ---------------------------------------  #

model=model3

python tests/grid_search/stream.py -m \
    mlflow.experiment_name=f_tuning_${model} \
    model=$model \
    model.h=2,3,4,5,6 \
    dark_mode=True \
    viz=False \
    save=True \
    verbose=False \
