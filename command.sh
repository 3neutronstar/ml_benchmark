#!/bin/bash
echo 'train lenet5'
conda activate bench
SET=$(seq 1 35)
for VALUE in $SET; do
    CUDA_VISIBLE_DEVICES=1 python run.py train --test_seed $VALUE
done
