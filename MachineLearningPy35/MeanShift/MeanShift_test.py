from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import decomposition
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from itertools import cycle

jfile = open("../data/measurement_data_40_L3_z.json", "r")
json_dict = json.loads(jfile.read())
############################################################################


def json_pull():
    """ Pull data from json file into a numpy matrix"""
    """
    train_mat will be the input matrix which we would like to add features to
    featrs will be a tuple of features we would like to add to the training matrix.
    """
    num_meas = 1000  # len(json_dict['measurements'])
    num_feat = 8

    # Define the training set
    datamat = np.matrix([[0.0 for col in range(num_feat)] for row in range(num_meas)])
    for i in range(num_meas):
        datamat[i, 0] = json_dict['measurements'][i]['data']['z']['time_domain_features']['p2p']
        datamat[i, 1] = json_dict['measurements'][i]['data']['z']['time_domain_features']['rms']
        datamat[i, 2] = json_dict['measurements'][i]['data']['z']['time_domain_features']['peak']
        datamat[i, 3] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_1x']
        datamat[i, 4] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_2x']
        datamat[i, 5] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_3x']
        datamat[i, 6] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_3x']
        datamat[i, 7] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_3x_to_10x_sum']

    nanarr = []
    for i in range(datamat.shape[0]):
        for j in range(num_feat):
            if math.isnan(datamat[i, j]):
                nanarr.append(i)
                break
    datamat = np.delete(datamat, nanarr, axis=0)
    return datamat

datamat = json_pull()

# #----------------------------------------------------------------------
# # Do Manual PCA
# print(datamat.shape)
# # Subtract out mean and normalize the data to standard deviation
# datamat -= np.mean(datamat, axis=0)
# datamat /= np.std(datamat, axis=0)
# cov_mat=np.cov(datamat, rowvar=False)
# evals, evecs = np.linalg.eigh(cov_mat)
# # print(evals)
#
# idx = np.argsort(evals)[::-1]
# print(idx)
# evecs = evecs[:,idx]
# evals = evals[idx]
#
# print(evecs)
# print(evecs[:2, :])
# variance_retained = np.cumsum(evals)/np.sum(evals)
# print(evals/np.sum(evals))


#----------------------------------------------------------------------
# Run PCA from SKLEARN
pca = decomposition.PCA(n_components=2)
pca.fit(datamat)
datamat = pca.transform(datamat)
print(pca.explained_variance_ratio_)


#----------------------------------------------------------------------
# Run MeanShift
bandwidth = estimate_bandwidth(datamat, quantile=0.2, n_samples=1000)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(datamat)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

print("cluster centers", cluster_centers)

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)


#----------------------------------------------------------------------
# Plotting
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
datamat = datamat
print(datamat.shape)

for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    # print("cluster_center", cluster_center)
    # print(cluster_center[0,0])
    # print(cluster_center[0][1])
    # print(col)

    plt.plot(datamat[my_members, 0], datamat[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

# #----------------------------------------------------------------------
# # Version with multiplication by a 2d eigenvector
# plt.figure(1)
# plt.clf()
#
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# datamat = datamat*np.transpose(evecs[:2, :])
# print(datamat.shape)
#
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = np.transpose(cluster_centers[k, :]*evecs[:2, :])
#     # print("cluster_center", cluster_center)
#     # print(cluster_center[0,0])
#     # print(cluster_center[0][1])
#     # print(col)
#
#     plt.plot(datamat[my_members, 0], datamat[my_members, 1], col + '.')
#     plt.plot(cluster_center[0, 0], cluster_center[0, 1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()
