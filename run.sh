#!/bin/bash

INIT=614125 # 85**3
FINAL=110592000 # 480 **3


# $1=gpu $2=datapath $3=expname
CUDA_VISIBLE_DEVICES=$1 python3 train.py --config configs/deblur_vm96.txt --datadir=$2 --expname=$3