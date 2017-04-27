from math import ceil
from scipy import *
import numpy as np
import matplotlib.pyplot as plt


def czt(x, m=None, w=None, a=None):
    # Translated from GNU Octave's czt.m

    n = len(x)
    if m is None:
        m = n
    if w is None:
        w = exp(-2j * pi / m)
    if a is None:
        a = 1

    chirp = w ** (arange(1 - n, max(m, n)) ** 2 / 2.0)
    N2 = int(2 ** ceil(log2(m + n - 1)))  # next power of 2
    xp = append(x * a ** -arange(n) * chirp[n - 1 : n + n - 1], zeros(N2 - n))
    ichirpp = append(1 / chirp[: m + n - 1], zeros(N2 - (m + n - 1)))
    r = ifft(fft(xp) * fft(ichirpp))
    return r[n - 1: m + n - 1] * chirp[n - 1: m + n - 1]


def get_signal(sampling_size, sampling_rate):
    t = 1.0 / sampling_rate
    x = np.linspace(0.0, sampling_size * t, sampling_size)
    y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    return y * np.hanning(sampling_size)


def plot_spectrum(y, sr, spectrum_type='fft'):
    ss = len(y)
    xf = np.linspace(0.0, sr/2, ss//2)
    print(len(xf))
    if spectrum_type == 'czt':
        yf = czt(y)
        yf = yf[0:ss//2]
    else:
        yf = np.fft.rfft(y)
        yf = yf[1:]

    yf *= 2.0
    plt.plot(xf, 2.0/ss * np.abs(yf), 'bo')
    plt.show()

plot_spectrum(get_signal(8192, 20000), 20000)
plot_spectrum(get_signal(10000, 20000), 20000)
plot_spectrum(get_signal(2 * 8192, 20000), 20000)
plot_spectrum(get_signal(20000, 20000), 20000)
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