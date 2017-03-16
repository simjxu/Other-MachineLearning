import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import sklearn

# Helpful pip:
# List outdated packages: pip list --outdated --format=columns
# Update outdated package: pip install --upgrade tensorflow

# Create a 2D matrix
x = np.matrix(((2, 3), (3, 5)))
y = np.matrix(((1, 2), (5, -1)))

# Show the size of the matrix
# print(x.shape[1])

# # Take individual items of the matrix squared
# TrainMatrix = np.matrix(((0, 0), (0, 1), (1, 0), (1, 1), (4, 4), (4, 5), (5, 4), (5, 5), (8, 8), (8, 9), (9, 8),
#                          (9, 9)))
# print(np.power(TrainMatrix,2))

# Creating a matrix of zeros
buckets = [[0 for col in range(5)] for row in range(10)]
#print(buckets)
bucketsm = np.matrix(buckets)
#print(bucketsm)
mat3 = np.matrix(np.ones((2,3)))
mat4 = np.matrix(np.ones((3,2)))
multi = mat3*mat4

# Make a 3d matrix, which is a list of 2d matrices
ThreeD = (np.matrix(((1, 1), (1, 1))), np.matrix(((2, 2), (2, 2))), np.matrix(((2, 2), (2, 2))))
part1 = ThreeD[0]
part2 = ThreeD[1]
ThreeD[1][1, 1] = 1
# print(ThreeD[1])
# multiplied12 = part1*part2
# print(multiplied12)

# Make a 3d matrix, using initialized ones
ThreeD = [np.matrix([[1 for col in range(2)] for row in range(2)]) for page in range(2)]
# print(ThreeD[0]*ThreeD[1])


#Phi = np.matrix(((0.33,0), (0.33,0), (0.34,0)))
Phi = np.matrix([[1. for col in range(1)] for row in range(3)])



#Sigma = np.matrix([[0 for col in range(2)] for row in range(3)])

# Sigma = np.zeros((2, 3, 4))
Sigma = np.matrix(np.zeros((2, 3)))
# print(Sigma)
#Sigma[1, 2] = 1
numclusters = 3
numfeatures = 2
Sigma = [[[0 for col in range(numfeatures)] for row in range(numfeatures)] for page in range(numclusters)]
for j in range(1):
    for i in range(2):
        Sigma[j][i][i] = 1

asdf = [[0 for col in range(3)] for row in range(3)]
for i in range(1):
    asdf[i][i] = 1
print(enumerate(asdf))
# # print(Sigma)
# km = KMeans(n_clusters=3)
# km.fit(TrainMatrix)
# centers = km.cluster_centers_
# Mu = np.matrix(centers)
# # print(Mu)

# iter = 0
# while iter < 10:
#     print(iter)
#     iter = iter+1

PDF = np.matrix(np.zeros((numclusters,1)))

newmat = np.matrix(np.zeros((2,1)))
newmat = Mu[0, :]
newmat2 = Mu[0, :]
Mu2 = Mu.tolist()
PDF[0] = multivariate_normal.pdf(TrainMatrix[0, :], mean=Mu2[0], cov=Sigma[0][:][:])


# Testing out PCA -------------------------------------
DataMatrix = np.matrix(((0, 0), (5, 1), (5, -1), (4, 1), (-3, 0)))

w, v = np.linalg.eig(np.diag((1, 2, 3)))
