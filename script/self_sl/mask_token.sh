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

python main.py train_mode=self_sl data=$data train_size=$train_size model=fttrans/mask_token model.trainer=FTTransMaskTokenSSLTrainer hydra.sweep.dir='outputs/self_sl/${train_size}/${data}/mask_token/${model.params.bias_after_mask}/${model.params.mask_ratio}/' seed="range(1,$max_seed)" model.params.mask_ratio=0.1,0.2,0.3,0.4,0.5,0.6,0.7 model.params.bias_after_mask=false -m
