#!/bin/bash
MODE="train_moo  baseline_moo";  
SET="0.4";
SPARSE_SET="1 2 3 4";
for VALUE in ${SET}
do
    for SPARSE_VALUE in ${SPARSE_SET}
    do
        for MODE_VAL in ${MODE}
        do
            python run.py $MODE_VAL --model vgg16 --moo_sparse_ratio $VALUE --moo_num_classes 5  --moo_num_sparse_classes $SPARSE_VALUE
        done
    done
done
