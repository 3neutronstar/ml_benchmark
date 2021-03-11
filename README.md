# ml_benchmark
benchmark repository

### Implementation
- Training Process
`python run.py train`

- Visualizing the gradient of weight for each layer
```shell script
    python run.py visual --file_name [date you train]
    tensorboard --logdir grad_data
```


- Checking the performance log
`tensorboard --logdir training_data`