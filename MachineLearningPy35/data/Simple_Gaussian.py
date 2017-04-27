import numpy as np
import json
import math
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
    num_feat = 500
    num_train = 300

    # Define the training set
    timestmps = ['a' for row in range(num_meas)]
    trainmatx = np.matrix([[0.0 for col in range(num_feat)] for row in range(num_meas)])
    for i in range(num_meas):
        timestmps[i] = json_dict['measurements'][i]['timestamp']
        # trainmatx[i, 0] = json_dict['measurements'][i]['data']['z']['time_domain_features']['rms']
        # trainmatx[i, 1] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_1x']
        # trainmatx[i, 2] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_2x']
        # trainmatx[i, 3] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_3x']
        # trainmatx[i, 4] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_3x_to_10x_sum']
        # trainmatx[i, 5] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_1x']
        # trainmatx[i, 6] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_2x']
        # trainmatx[i, 7] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_3x']
        for j in range(num_feat):
            trainmatx[i, j] = json_dict['measurements'][i]['data']['z']['frequency_domain']['amps'][j]

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
    stdev = np.std(trainmatx, axis=0)
    for i in range(stdev.shape[1]):
        if stdev[0, i] == 0.:
            zerostdarr.append(i)

    trainmatx = np.delete(trainmatx, zerostdarr, axis=1)

    print("Number of measurements:", trainmatx.shape[0], "Number of features:", trainmatx.shape[1])
    print("Number of timestamps:", len(timestmps))

    # Define the test set. BE CAREFUL, A SEPARATE COPY OF THE MATRIX IS NOT CREATED, TRAIN_X AND TEST_X REFERENCE SAME
    train_X = np.asmatrix(np.copy(trainmatx[0:num_train, :]))
    test_X = np.asmatrix(np.copy(trainmatx[0:1000, :]))
    train_y = train_X
    test_y = test_X

    return train_X, test_X, train_y, test_y, timestmps

def main():

    train_X, test_X, train_y, test_y, timestmps = json_pull()
    num_feat = test_X.shape[1]
    num_meas = test_X.shape[0]

    std = np.std(train_X[:,1:300], axis=0)
    print(std.shape[1])
    for i in range(std.shape[1]):
        if std[i, 0] == 0:
            print("found zero")
            print(i)


    diff_mat = (test_X-np.mean(train_X, axis=0))/np.std(train_X, axis=0)

    scoreval = [0. for row in range(num_meas)]
    for i in range(num_meas):
        feat_score = 0
        for j in range(num_feat):
            if diff_mat[i, j] > 0:
                feat_score = feat_score + diff_mat[i, j]
        scoreval[i] = feat_score

    plt.plot(scoreval, 'bo')
    plt.show()


if __name__ == '__main__':
    main()