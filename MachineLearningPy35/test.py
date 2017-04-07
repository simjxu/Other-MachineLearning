import matplotlib.pyplot as plt
import numpy as np
A =

#-----------------------------------------------------------------------------------------------
# import tensorflow as tf
#
# # Creates a graph.
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))


#-----------------------------------------------------------------------------------------------
# import numpy as np
# from sklearn.cluster import MeanShift, estimate_bandwidth
# from sklearn.datasets.samples_generator import make_blobs
# import matplotlib.pyplot as plt
# from itertools import cycle
#
# def PCA(DataMatrix):
#     numsamples = DataMatrix.shape[0]
#     numfeatures = DataMatrix.shape[1]
#
#     # DataMatrixnorm = [[0 for col in range(numfeatures)] for row in range(numsamples)]
#     DataMatrixnorm = np.matrix(np.zeros((numsamples, numfeatures)))
#     # Zero out the mean and scale variance -------------------------
#
#     # Zero out mean
#     mu = np.matrix(np.zeros((numfeatures, 1)))
#     for j in range(numfeatures):
#         mu[j] = np.sum(DataMatrix[:, j])/numsamples
#     for i in range(numsamples):
#         DataMatrixnorm[i, :] = DataMatrix[i, :]-np.transpose(mu)
#
#     # Set to unit variance
#     for j in range(numfeatures):
#         sigma = np.sqrt(np.sum(np.power(DataMatrix[:, j], 2))/numsamples)
#         if sigma == 0.0:
#             DataMatrixnorm[:, j] = np.matrix(np.zeros((numsamples, 1)))
#         else:
#             for i in range(numsamples):
#                 DataMatrixnorm[i, j] = DataMatrixnorm[i, j]/sigma
#
#     # Calculate Covariance
#     # Covar = [[0 for col in range(numfeatures)] for row in range(numfeatures)]
#     Covar = np.matrix(np.zeros((numfeatures, numfeatures)))
#     for i in range(numsamples):
#         Covar += np.transpose(DataMatrixnorm[i, :])*DataMatrixnorm[i, :]
#     Covar *= 1/numsamples
#
#     # Identify the eigenvectors and eigenvalues. The largest eigenvalues correspond to the eigenvector with the largest
#     # variance.
#     eigval, eigvec = np.linalg.eig(Covar)
#
#     maxidx = np.argmax(eigval)
#
#     primaryvec = eigvec[:, maxidx]
#
#     return eigval, eigvec, primaryvec
#
# centers = [[1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, -1], [1, -1, 1, -1, 1, -1, 1, -1]]
# X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)
#
# # The following bandwidth can be automatically detected using
# bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
#
# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# ms.fit(X)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_
#
# labels_unique = np.unique(labels)
# n_clusters_ = len(labels_unique)
#
# eigval, eigvec, primaryvec = PCA(datamat)
#
# vec2d = eigvec[:, :2]
# X = X*vec2d
#
# print("number of estimated clusters : %d" % n_clusters_)
#
# plt.figure(1)
# plt.clf()
#
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = cluster_centers[k, :]*vec2d
#     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()