import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

# Create a 2D matrix
x = np.matrix(((2, 3), (3, 5)))
y = np.matrix(((1, 2), (5, -1)))

# Show the size of the matrix
# print(x.shape[1])

# Training Matrix (will have to convert a .csv file to matrix
TrainMatrix = np.matrix(((0, 0), (0, 1), (1, 0), (1, 1), (4, 4), (4, 5), (5, 4), (5, 5), (8, 8), (8, 9), (9, 8),
                         (9, 9)))

# Declare an array of ones

# Make guesses on the Mu, Sigma, and Phi. Mu can be guessed using k-means
Mu = np.matrix(((0, 0), (4,4), (8,8)))
Sigma = (np.matrix(((1, 0.1), (0.1, 1))), np.matrix(((1, 0.1), (0.1, 1))), np.matrix(((1, 0.1), (0.1, 1))))
Phi = np.matrix((0.33, 0.33, 0.34))

numiterations = 10




def Kmeans_MixGauss(TrainMatrix, numclusters, numiterations):

    # Identify number of features and number of training examples
    numtrainexamples = TrainMatrix.shape[0]
    numfeatures = TrainMatrix.shape[1]

    # Set defaults for initial guesses -----------------------------
    # Assume same prob for all clusters
    Phi = np.matrix([[1 for col in range(1)] for row in range(numclusters)])

    # Assume a circular unit standard deviation (will need to use np.matrix when doing matrix math)
    Sigma = [np.matrix([[0 for col in range(numfeatures)] for row in range(numfeatures)]) for page in range(numclusters)]
    for j in range(numclusters):
        for i in range(numfeatures):
            Sigma[j][i, i] = 1             # Sigma is a list of 2D matrices, so need to use a bracket with comma

    # First, use kmeans to find the cluster centroids, and use those for means
    km = KMeans(n_clusters=numclusters)         # Indicate number of clusters for Kmeans
    km.fit(TrainMatrix)
    TMcenters = km.cluster_centers_
    Mu = np.matrix(TMcenters)                   # Make it into a matrix

    # Begin while loop (following procedure in page 2-3 of the Mixture of Gaussians and EM algorithm section
    iter = 0
    w = np.matrix(np.zeros((numclusters, numtrainexamples)))
    print(w)
    while iter < numiterations:

        # E-step: determine the w-array
        PDF = np.matrix(np.zeros((numclusters, 1)))         # Matrices need double parentheses.
        for i in range(numtrainexamples):
            # E-step
            denom = 0
            for n in range(numclusters):
                # Need to figure this part out!! mean= prefers a list as opposed to a matrix for some reason
                Mulist = Mu.tolist()                        # Convert Mu into a list first
                PDF[n] = multivariate_normal.pdf(TrainMatrix[i, :], mean=Mulist[n], cov=Sigma[n])*Phi[n]
                denom = denom+PDF[n]

            # Calculate the w probability for each cluster
            w[:, i] = PDF/sum(PDF)

        # Zero out the variables
        Mu = [[0 for col in range(numfeatures)] for row in range(numclusters)]
        Sigma = [[[0 for col in range(numfeatures)] for row in range(numfeatures)] for page in range(numclusters)]

        # M-step: Update phi for each category
        for n in range(numclusters):

            Phi[n] = np.sum(w[n, :])/numtrainexamples
            for i in range(numtrainexamples):
                Mu[n, :] = Mu[n, :] + w[n, i]*TrainMatrix[i, :]
                Sigma[n] = Sigma[n] + w[:, i]*np.transpose(TrainMatrix[i, :]-Mu[n, :])*(TrainMatrix[i, :]-Mu[n, :])

            Mu[n, :] = np.divide(Mu[n, :], sum(w[n, :]))    # Is this how you divide?
            Sigma[n] = np.divide(Sigma[n], sum(w[n, :]))    # Is this how you divide?

        iter += 1

    return Mu, Sigma, Phi


newMu, newSigma, newPhi = Kmeans_MixGauss(TrainMatrix, 3, numiterations)