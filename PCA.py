import numpy as np


DataMatrix = np.matrix(((0, 0), (-1, -1), (-2, -2), (1, 1), (2, 2)))

def PCA(DataMatrix):
    numsamples = DataMatrix.shape[0]
    numfeatures = DataMatrix.shape[1]

    # DataMatrixnorm = [[0 for col in range(numfeatures)] for row in range(numsamples)]
    DataMatrixnorm = np.matrix(np.zeros((numsamples, numfeatures)))
    # Zero out the mean and scale variance -------------------------

    # Zero out mean
    mu = np.matrix(np.zeros((numfeatures, 1)))
    for j in range(numfeatures):
        mu[j] = np.sum(DataMatrix[:, j])/numsamples
    for i in range(numsamples):
        DataMatrixnorm[i, :] = DataMatrix[i, :]-np.transpose(mu)

    # Set to unit variance
    for j in range(numfeatures):
        sigma = np.sqrt(np.sum(np.power(DataMatrix[:,j],2))/numsamples)
        for i in range(numsamples):
            DataMatrixnorm[i, :] = DataMatrix[i, j]/sigma

    # Calculate Covariance
    # Covar = [[0 for col in range(numfeatures)] for row in range(numfeatures)]
    Covar = np.matrix(np.zeros((numfeatures, numfeatures)))
    for i in range(numsamples):
        Covar += np.transpose(DataMatrixnorm[i, :])*DataMatrixnorm[i, :]
    Covar *= 1/numsamples

    # Identify the eigenvectors and eigenvalues. The largest eigenvalues correspond to the eigenvector with the largest
    # variance.
    eigval, eigvec = np.linalg.eig(Covar)

    maxidx = np.argmax(eigval)

    primaryvec = eigvec[:, maxidx]

    return eigval, eigvec, primaryvec

caleigval, caleigvec, calpvec = PCA(DataMatrix)
print(calpvec)
