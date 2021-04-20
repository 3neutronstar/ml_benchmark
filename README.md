# ml_benchmark
benchmark repository

### Implementation
- Training Process 
`python run.py train`

- Choose Algorithm
```shell script
    python run.py train --nn_type [lenet5 lenet300_100 vgg16]
```
- Training classification with multi-task learning (v1 and v3 is gpu/cpu using mtl method, v2 and v4 is using classwise sorted dataset)
```shell script
    python run.py train_mtl[ ,_v2,_v3,v_4] --nn_type [neural net]
```
v2 and v4 should not be use neural net that use BatchNorm

- Visualizing the gradient of weight for each layer
```shell script
    python run.py visual --file_name [date you train]
    tensorboard --logdir grad_data/[data you train]
```
- Training the gradient lateral inhibition with threshold
```shell script
    python run.py train_grad_prune --nn_type vgg16    
```

- Pruning weight and gradient while training by cumulative value of gradient (vgg16 is available, but not recommended)
```shell script
    python run.py train_weight_prune --threshold 100 -- grad_off_epoch 5 --nn_type [lenet5 or lenet300_100]
```
- Running in Colab
```shell script
    python run.py train --colab True
```
- Checking the performance log
`tensorboard --logdir training_data`
