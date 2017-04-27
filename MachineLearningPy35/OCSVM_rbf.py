import numpy as np
import json
import math
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

from itertools import product

def json_pull():
    """ Pull data from json file into a numpy matrix"""
    """
    train_mat will be the input matrix which we would like to add features to
    featrs will be a tuple of features we would like to add to the training matrix.
    """
    # Get JSON File
    jfile = open("C:\\Users\\Simon\\Documents\\Data\\measurement_data_40_L3_z.json", "r")
    json_dict = json.loads(jfile.read())

    num_meas = len(json_dict['measurements'])
    num_feat = 2

    # Define the training set
    timestmps = ['a' for row in range(num_meas)]
    trainmatx = np.matrix([[0.0 for col in range(num_feat)] for row in range(num_meas)])
    for i in range(num_meas):
        timestmps[i] = json_dict['measurements'][i]['timestamp']
        # trainmatx[i, 0] = json_dict['measurements'][i]['data']['z']['time_domain_features']['rms']
        trainmatx[i, 0] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_1x']
        # trainmatx[i, 2] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_2x']
        # trainmatx[i, 3] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_3x']
        trainmatx[i, 1] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_1x']
        # trainmatx[i, 5] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_2x']
        # trainmatx[i, 6] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_3x']
        # trainmatx[i, 7] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_3x_to_10x_sum']
        # for j in range(num_feat):
        #     trainmatx[i, j] = json_dict['measurements'][i]['data']['y']['frequency_domain']['amps'][j]

    # Sort timestamp list and also get the sort indices
    sortidx = [i[0] for i in sorted(enumerate(timestmps), key=lambda x: x[1])]
    timestmps.sort()
    print("first time:", timestmps[0])

    # Sort training matrix by indices dictated by timestamps
    trainmatx = trainmatx[sortidx]

    # Identify and remove all the NaNs in the matrix
    nanarr = []
    for i in range(trainmatx.shape[0]):
        for j in range(num_feat):
            if math.isnan(trainmatx[i, j]):
                nanarr.append(i)
                break
    trainmatx = np.delete(trainmatx, nanarr, axis=0)
    timestmps = [i for j, i in enumerate(timestmps) if j not in nanarr]

    # Identify and remove all columns that have the same value
    zerostdarr = []
    for i in range(trainmatx.shape[1]):
        for j in range(trainmatx.shape[0]):
            if j != 0 and trainmatx[j, i] == trainmatx[j-1, i]:
                if j == num_meas-1:
                    zerostdarr.append(i)
            elif trainmatx[j, i] != trainmatx[j-1, i]:
                break
    trainmatx = np.delete(trainmatx, zerostdarr, axis=1)

    print("Number of measurements:", trainmatx.shape[0], "Number of features:", trainmatx.shape[1])
    print("Number of timestamps:", len(timestmps))

    # Define the test set. BE CAREFUL, A SEPARATE COPY OF THE MATRIX IS NOT CREATED, TRAIN_X AND TEST_X REFERENCE SAME
    train_X = np.copy(trainmatx[0:10, :])
    test_X = np.copy(trainmatx[0:1050, :])
    train_y = train_X
    test_y = test_X
    return train_X, test_X, train_y, test_y, timestmps

def normalize(train_matrix):
    """ Normalize the training matrix (and also set to unit variance?) """

    train_matrix -= np.mean(train_matrix, axis=0)
    train_matrix /= np.std(train_matrix, axis=0)

    return train_matrix

def main():
    train_X, test_X, train_y, test_y, timestmps = json_pull()

    train_X = normalize(train_X)
    test_X = normalize(test_X)

    num_meas = test_X.shape[0]
    model = OneClassSVM(nu=0.1, kernel="rbf", gamma='auto')
    model.fit(train_X)

    fits = model.fit(train_X)
    print("fits", fits)

    scoreval = [0. for row in range(num_meas)]
    for i in range(num_meas):
        curr_meas = np.matrix(test_X[i, :])
        scoreval[i] = np.matrix.item(model.decision_function(curr_meas))

    # # Plotting decision regions
    # x_min, x_max = test_X[:, 0].min() - 1, test_X[:, 0].max() + 1
    # y_min, y_max = test_X[:, 1].min() - 1, test_X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
    #                      np.arange(y_min, y_max, 0.1))
    #
    # f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))
    #
    # for idx, clf, tt in zip(product([0, 1], [0, 1]),
    #                         [model, model, model, model],
    #                         ['One Class SVM', 'KNN (k=7)',
    #                          'Kernel SVM', 'Soft Voting']):
    #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #     Z = Z.reshape(xx.shape)
    #
    #     axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.5)
    #     axarr[idx[0], idx[1]].scatter(test_X[:, 0], test_X[:, 1], alpha=0.4)
    #     axarr[idx[0], idx[1]].set_title(tt)

    plt.plot(scoreval, 'bo')
    plt.show()

if __name__ == '__main__':
    main()
