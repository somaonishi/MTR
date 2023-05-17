#!/bin/bash

cd "$(dirname "$0")"
cd ../..
source .venv/bin/activate
data=$1
train_size=$2

if [ -z "$3" ]; then
  max_seed=11
else
  max_seed=$(($3 + 1))
fi

echo $data $train_size $max_seed

python main.py train_mode=self_sl data=$data train_size=$train_size model=fttrans/hidden_mix model.trainer=FTTransHiddenMixSSLTrainer hydra.sweep.dir='outputs/self_sl/${train_size}/${data}/hidden_mix/${model.params.alpha}/' seed="range(1,$max_seed)" model.params.alpha=0.1,0.2,0.5,0.75,1.0,1.5,2.0 -m
