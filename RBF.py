import numpy as np
from linearRegression import linearRegression
from kMeans import kmeans
class RBF:
    
    def __init__(self,  numBases, epoches, learningRate, is_dynamic) -> None:
        self.epoches = epoches
        self.learningRate = learningRate
        self.numBases = numBases
        self.linear_regression = linearRegression(numBases)
        self.kmeans = kmeans(self.numBases, is_dynamic)

    def getGaussians(self, data):
        res = np.zeros((data.shape[0], self.numBases))
        # calculate exp(-1/2sigma^2 * ||x-xj||^2) 
        for i, j in enumerate(data):
            for k, (center, variance) in enumerate(zip(self.kmeans.centroids, self.kmeans.varriances)):
                res[i, k] = np.exp((-1 / (2 * variance)) * (np.square(np.linalg.norm(j - center))))
        return res
  
    def approximation(self, data):
        # F(x) = summation w_j * Gaussian + bias
        g = self.getGaussians(data)
        return np.dot(g,self.linear_regression.weight)+self.linear_regression.bias
    
    def get_sse(self, x, hx):
        return np.sum(np.square(hx - self.approximation(x)))
    
    def train(self, data, desired):
       self.kmeans.process_kmeans(data)
       res = self.getGaussians(data)
       self.linear_regression.linear_regression(res,desired,self.learningRate,self.epoches)