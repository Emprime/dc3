#!/usr/bin/env bash

# run in docker container

export ML_DATA=/data-ssd/fixmatch/
export PYTHONPATH=$PYTHONPATH:.

declare -a StringArray=('cifar10h' 'plankton' 'turkey' 'miceBone')

# create labeled datasets
# if they already exists you need to manually delete them
CUDA_VISIBLE_DEVICES=0 ./scripts/create_datasets.py

# generate unlabeled data
for dataset in "${StringArray[@]}"; do
    CUDA_VISIBLE_DEVICES=0 ./scripts/create_unlabeled.py $ML_DATA/SSL2/$dataset $ML_DATA/$dataset-train.tfrecord $ML_DATA/$dataset-unlabeled.tfrecord ;
done
wait

# generate splits
for dataset in "${StringArray[@]}"; do
    CUDA_VISIBLE_DEVICES=0 ./scripts/create_split.py --seed=1 --size=-1 $ML_DATA/SSL2/$dataset $ML_DATA/$dataset-train.tfrecord;
done
wait