#!/bin/bash
MODE="train_lbl_v2  baseline_moo";  
SET="0.1 0.2 0.3 0.4";
SPARSE_SET="1 2 3 4";
for VALUE in ${SET}
do
    for SPARSE_VALUE in ${SPARSE_SET}
    do
        for MODE_VAL in ${MODE}
        do
            python run.py $MODE_VAL --model resnet20 --moo_sparse_ratio $VALUE --moo_num_classes 10  --moo_num_sparse_classes $SPARSE_VALUE
        done
    done
done
