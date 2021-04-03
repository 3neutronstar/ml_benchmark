# ml_benchmark
benchmark repository

### Implementation
- Training Process 
`python run.py train`

- Choose Algorithm
```shell script
    python run.py train --nn_type [lenet5 lenet300_100 vgg16]
```
- Visualizing the gradient of weight for each layer
```shell script
    python run.py visual --file_name [date you train]
    tensorboard --logdir grad_data/[data you train]
```
- Visualizing the gradient prune with threshold
```shell script
    python run.py visual_prune --threshold 100
```

- Pruning weight and gradient while training by cumulative value of gradient (vgg16 is available, but not recommended)
```shell script
    python run.py train_prune --threshold 100 -- grad_off_epoch 5 --nn_type [lenet5 or lenet300_100]
```
- Running in Colab
```shell script
    python run.py train --colab True
```
- Checking the performance log
`tensorboard --logdir training_data`
