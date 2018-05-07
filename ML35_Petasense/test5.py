import numpy as np
from sklearn.ensemble import IsolationForest
import json
import matplotlib.pyplot as plt

# Features: Acceleration/Velocity RMS (2*3 = 6)
# Spectrum band energy velocity of 100 Hz up to 500Hz (5*3 = 15)
# Spectrum band energy ancceleration of 500Hz up to 1500Hz (2*3 = 6)
# Total features = 27 features

# Download the json file
def json_pull(filename):
    """ Pull data from json file into a numpy matrix"""
    """
    train_mat will be the input matrix which we would like to add features to
    featrs will be a tuple of features we would like to add to the training matrix.
    """
    # Get JSON File
    jfile = open("C:\\Users\\Simon\\Downloads\\" + filename, "r")
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
    X = np.copy(trainmatx[0:1050, :])

    return X

# Pull out the features, sum it over


X_baseline = json_pull("raw_data.json")             # Encompasses dates Feb 10 2017 4am to Mar 22 7pm 2017
X = json_pull("raw_data2.json")



# Fit model on Isolation forest
clf = IsolationForest(max_sample=100 random_stage=rng)
clf.fit(X_baseline)
dec_func = clf.decision_function(X)
plt.plot(dec_func)
plt.show()