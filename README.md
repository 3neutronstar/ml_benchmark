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
    tensorboard --logdir grad_data
```


- Checking the performance log
`tensorboard --logdir training_data`
