import numpy as np
from Learner.base_learner import BaseLearner
class NumpyLearner(BaseLearner):
    def __init__(self, model, time_data, file_path, configs):
        super(NumpyLearner,self).__init__(model, time_data, file_path, configs)

    def run(self):
        return