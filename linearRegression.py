import numpy as np
class linearRegression():
    def __init__(self, K):
        self.bias =0
        self.weight = np.zeros((K, 1))
        
    def linear_regression(self, data, desired, learningRate, epoches):
        for i in range(epoches):
                for k, j in zip(data, desired):
                    k = k.reshape(data.shape[1], 1)
                    error = j - (np.dot(k.T, self.weight) + self.bias)
                    # update weight
                    self.weight += learningRate * error * k
                    # update bias
                    self.bias += learningRate * error       