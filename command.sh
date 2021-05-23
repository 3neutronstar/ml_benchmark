#!/bin/bash
echo 'train lenet5'
conda activate bench
MODE=$(train_lbl, baseline_moo)
SET=$(0.1,0.2,0.3,0.4)
SPARSE_VALUE=$(1,2,3)
for VALUE in $SET; 
do
    for SPARSE_VALUE in $SPARSE_SET
    do
        for MODE_VAL in $MODE
        do
            python run.py $MODE_VAL --model vgg16 --moo_sparse_ratio $VALUE --moo_num_classes 5 --moo_num_sparse_classes $SPARSE_VALUE
        done
    done
done