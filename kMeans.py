import numpy as np
import math
class kmeans():
    def __init__(self,  k, is_dynamic) -> None:
        self.k = k
        self.centroids = np.zeros(k)
        self.varriances = np.zeros(k)
        self.is_dynamic = is_dynamic
        self.sse = 0

    #calculate the Euclidean distance of two points
    # input: x1 double, x2 double
    # output: the Euclidean distance double
    def Euclidean_dis(self, x1, x2):
        return math.sqrt((x1 - x2) ** 2)
        
    # Using k means algorithm to find k clusters
    # input: k: number of clusters, data: data set
    # output: k clustered center and variances
    def process_kmeans(self, data):
        # step1 select K cluster centers randomly
        self.centroids = data[np.random.choice(range(75), self.k, replace=False)]
        converge = False 
        while(not converge):
            clusters = [[]for i in range(len(self.centroids))]
                
        # step2: find the cluster for each point
            for i in data:
                globalMinIndex = 0
                globalMin = self.Euclidean_dis(i, self.centroids[0])
                for j in range(1, len(self.centroids)):
                    curDist = self.Euclidean_dis(i, self.centroids[j])
                    if curDist < globalMin:
                        globalMin = curDist
                        globalMinIndex = j
                clusters[globalMinIndex].append(i)
                
            # step3: update cluster centers
            clusters = list((filter(None, clusters)))
            prevCenter = self.centroids.copy()
            self.centroids = []
            for i in range(len(clusters)):
                self.centroids.append(np.mean(clusters[i],axis =0))
            pattern = np.abs(np.sum(prevCenter) - np.sum(self.centroids))
            #check if the cluster center changed 
            converge = (pattern == 0)
            
        # check if the cluster only contains 1 point
        if(self.is_dynamic):
            self.dynamic(clusters)
        else:
            self.same(self.centroids)
            
    def dynamic(self, clusters):
        normalClusterVar =[]
        onePointClusterPos =[]
        for i in range(len(clusters)):
            if len(clusters[i]) >1:
                var = np.var(clusters[i])
                self.varriances[i] = var
                normalClusterVar.append(var)
            else:
                onePointClusterPos.append(i)
        for i in range(len(onePointClusterPos)):
           self.varriances[onePointClusterPos[i]] = np.mean(normalClusterVar)
           
    def same(self, centroids):
        max =0
        for i in range(len(centroids)):
            for j in range(len(centroids)):
                cur_dist = np.linalg.norm(centroids[i]-centroids[j])
                if cur_dist > max:
                    max = cur_dist
        self.varriances= np.ones(self.k) * np.square(max/np.sqrt(2 * self.k))