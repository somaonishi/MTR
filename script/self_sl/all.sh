#!/bin/bash

cd "$(dirname "$0")"

data=$1
train_size=$2

if [ -z "$3" ]; then
  max_seed=10
else
  max_seed=$3
fi

echo $data $train_size $max_seed

./wo-da.sh $data $train_size $max_seed
./mask_token.sh $data $train_size $max_seed
./hidden_mix.sh $data $train_size $max_seed
./mixup.sh $data $train_size $max_seed
./scarf.sh $data $train_size $max_seed
