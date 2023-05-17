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

python main.py train_mode=supervised data=$data train_size=$train_size hydra.sweep.dir='outputs/self_sl/${train_size}/${data}/wo-da/' seed="range(1,$max_seed)" -m

