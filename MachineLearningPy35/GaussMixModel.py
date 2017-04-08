import numpy as np
import json
import math
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

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
    num_feat = 8

    # Define the training set
    timestmps = ['a' for row in range(num_meas)]
    trainmatx = np.matrix([[0.0 for col in range(num_feat)] for row in range(num_meas)])
    for i in range(num_meas):
        timestmps[i] = json_dict['measurements'][i]['timestamp']
        trainmatx[i, 0] = json_dict['measurements'][i]['data']['z']['time_domain_features']['rms']
        trainmatx[i, 1] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_1x']
        trainmatx[i, 2] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_2x']
        trainmatx[i, 3] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_3x']
        trainmatx[i, 4] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_1x']
        trainmatx[i, 5] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_2x']
        trainmatx[i, 6] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_3x']
        trainmatx[i, 7] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_3x_to_10x_sum']
        # for j in range(num_feat):
        #     trainmatx[i, j] = json_dict['measurements'][i]['data']['z']['frequency_domain']['amps'][j]

    # Sort timestamp list and also get the sort indices
    sortidx = [i[0] for i in sorted(enumerate(timestmps), key=lambda x: x[1])]
    timestmps.sort()
    print("first time:", timestmps[0])

    # Sort training matrix by indices dictated by timestamps
    trainmatx = trainmatx[sortidx]
    print("first rms:", trainmatx[0, 0])

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
    train_X = np.copy(trainmatx[0:200, :])
    test_X = np.copy(trainmatx[0:1050, :])
    train_y = train_X
    test_y = test_X
    return train_X, test_X, train_y, test_y, timestmps

def main():

    num_clusters = 1
    num_iterations = 100

    train_X, test_X, train_y, test_y, timestmps = json_pull()

    num_meas = test_X.shape[0]
    model = GaussianMixture(n_components=num_clusters, max_iter=num_iterations)
    model.fit(train_X)

    GM_means = model.means_
    GM_stds = model.covariances_

    print(GM_means)

    diag = np.matrix([0. for row in range(GM_stds.shape[1])])
    scoreval = [0. for row in range(num_meas)]
    for i in range(num_meas):
        curr_meas = np.matrix(test_X[i, :])

        # Identify which cluster each measurement belongs to
        pred_cluster = model.predict(curr_meas)
        for j in range(GM_stds.shape[1]):
            diag[0, j] = GM_stds[pred_cluster, j, j]

        # See how much it deviates from that cluster mean
        #dev_arr = (curr_meas - GM_means[pred_cluster, :])*np.linalg.inv(GM_stds[pred_cluster,:,:])
        dev_arr = (curr_meas - GM_means[pred_cluster, :])/diag

        dev_arr = np.squeeze(np.asarray(dev_arr))
        # Output the score, decide on max or mean, or Mahalanobis distance
        # scoreval[i] = np.ndarray.max(dev_arr)
        p_or_n = np.ndarray.mean(dev_arr)/500
        # scoreval[i] = model.score(curr_meas)
        scoreval[i] = np.matrix.item(np.sqrt((curr_meas-GM_means[pred_cluster, :])
                               * np.linalg.inv(GM_stds[pred_cluster, :, :])
                               * np.transpose(curr_meas-GM_means[pred_cluster, :])))*p_or_n      # Mahalanobis distance

        if i == 748:
            print("This is", i)
            print("curr_meas:", curr_meas)
            print("GMMeans:", GM_means[pred_cluster, :])
            print("p_or_n:", p_or_n)
            print("covar multiply:", (curr_meas-GM_means[pred_cluster, :])*np.linalg.inv(GM_stds[pred_cluster, :, :])
                  *np.transpose(curr_meas-GM_means[pred_cluster, :]))
            print("stdevs:", diag)
            print(dev_arr)

        # if i == 58:
        #     print("This is", i)
        #     print("curr_meas:", curr_meas)
        #     print("GMMeans:", GM_means[pred_cluster, :])
        #     print("difference:", curr_meas - GM_means[pred_cluster, :])
        #
        #     print("stdevs:", diag)
        #     print(dev_arr)


    plt.plot(scoreval, 'bo')
    plt.show()

    inpval = int(input('Enter measurement number:'))
    print(np.matrix((test_X[inpval, :]-GM_means[0, :]))*np.linalg.inv(GM_stds[0, :, :]))
    print('Feature number', np.argmax((np.matrix(test_X[inpval, :])-GM_means[pred_cluster, :])
                                      *np.linalg.inv(GM_stds[pred_cluster, :, :])),
          'contributed deviation value of', np.max((np.matrix(test_X[inpval, :])-GM_means[pred_cluster, :])
                                      *np.linalg.inv(GM_stds[pred_cluster, :, :])))

if __name__ == '__main__':
    main()
