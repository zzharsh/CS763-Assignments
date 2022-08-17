import os
import random
import numpy as np
import matplotlib.pyplot as plt
# * YOU ARE NOT ALLOWED TO IMPORT ANYTHING ELSE  *#

seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

# Here is a template for the KMeans clustering class
# The class has two functions "fit" and "predict" that you need to implement
# The fit function generates and stores  cluster centers for the
# given data
# The predict function assigns the input data to clusters
# (as learned by the fit function)
#  The input X should be a numpy array of of dimensions (n_data, n_features)
#  n_data is the dataset size
#  n_features is the feature dimension. For example for 3D datapoints,
#  n_features will be 3
#  Do not hard code n_features to any value - We will test your code on
# high dimension data


class KMeans(object):
    def __init__(self, n_clusters):
        super(KMeans, self).__init__()
        self.n_clusters = n_clusters
        self.centers = None

    def fit(self, X, iterr= 1000):
        epsilon = 0
        k = self.n_clusters
        it = 0
        if self.centers is None:
            self.centers = np.random.normal(0,0.1, size = (k,X.shape[1]))
        
        while True:
            # Assign data points to clusters
            label = np.zeros((X.shape[0],1))
            distances = np.zeros((X.shape[0],k))
            
            for i in range(k):
                distances[:,i] = np.linalg.norm(X-self.centers[i], axis = 1)
            label = np.argmin(distances, axis=1)

            # Update cluster centers
            old_c = self.centers # will help in break clause
            for i in range(k):
                
                self.centers[i] = np.mean(X[label == i],axis=0)

            # Breaking condition
            if np.linalg.norm(old_c-self.centers)< epsilon or it> iterr:
                break
            old_c = self.centers.copy()
            it += 1;
                


    # Returns a numpy array of dimensions (n_data,)
    def predict(self,X):
        # Assign data points to clusters
        distances = np.zeros((X.shape[0],self.n_clusters))
        for i in range(self.n_clusters):
            distances[:,i] = np.linalg.norm(X-self.centers[i], axis = 1)
        label = np.argmin(distances, axis=1)
        return label


# Feel free to experiment with this


NUM_CLUSTERS = 3

# Create data such that it naturally forms clusters in 2D space 
# shape of this numpy array should be (n_data, n_features)
# here n_data is the dataset size (should be 600)
# here n_feautures is the dataset size (should be 2)
X = np.zeros((600,2))
mean1 = [3,2]
mean2 = [1,5]
mean3 = [0,1]
cov1 = [[0.5, 0],[0, 0.5]]
cov2 = [[0.5, 0],[0,0.5]]
cov3 = [[0.5,0],[0,0.5]]
X[:200,:] = np.random.multivariate_normal(mean1, cov1, 200)
X[200:400,:] = np.random.multivariate_normal(mean2, cov2, 200)
X[400:,:] = np.random.multivariate_normal(mean3, cov3, 200)
dataIn = X

# show data
plt.figure()
plt.scatter(dataIn[:,0], dataIn[:,1], alpha = 0.3, marker = 'o')
plt.title('Input Data')
plt.savefig('../results/inputCluster.png')

kmm = KMeans(n_clusters = NUM_CLUSTERS)
kmm.fit(dataIn)
preds = kmm.predict(dataIn)
plt.figure()
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'lime', 'turquoise', 'blueviolet', 'crimson', 'peru', 'maroon']
for ci in range(NUM_CLUSTERS):
    indices = preds == ci
    plt.scatter(dataIn[indices,0],dataIn[indices,1], alpha = 0.3, marker = 'o', color = colors[ci%len(colors)], label = 'Cluster {}'.format(ci))
    plt.legend()
    plt.title('Clustered Data')
    plt.savefig('../results/outputCluster.png')
