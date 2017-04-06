"""
Autoencoder can be used as a replicator neural network
http://neuro.bstu.by/ai/To-dom/My_research/Paper-0-again/For-research/D-mining/Anomaly-D/KDD-cup-99/NN/dawak02.pdf
Use a simple feedforward neural network with a single hidden layer to predict the class
N input nodes ---> hidden layer ---> hidden layer ---> .... ---> N output nodes
"""
import tensorflow as tf
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from time import gmtime, strftime

# Enter number of nodes in each fully connected layer
# h_size is a list that holds the number of nodes in each layer
h_size = [30, 3, 30]
num_epochs = 2500

############################################################################
# Change the following function to get the data that you are using

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
    train_X = np.copy(trainmatx[0:400, :])
    test_X = np.copy(trainmatx[0:1050, :])
    train_y = train_X
    test_y = test_X
    return train_X, test_X, train_y, test_y, timestmps

def select_features(feature_str):
    """ 4097 frequency bins in the frequency array """
    num_feat = len(feature_str)

def normalize(train_matrix):
    """ Normalize the training matrix (and also set to unit variance?) """

    # t = np.arange(1, train_matrix.shape[0] + 1, 1)
    # plt.plot(t, train_matrix[:, 0], 'r-')
    train_matrix -= np.mean(train_matrix, axis=0)
    train_matrix /= np.std(train_matrix, axis=0)
    # plt.plot(t, train_matrix[:, 0], 'b-')
    # plt.show()

    return train_matrix

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(A, B):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    # The sigmoid function
    h    = tf.nn.sigmoid(tf.matmul(A, B))

    # # Alternative function using ReLU
    # h    = tf.nn.relu(tf.matmul(A, B))
    return h

def main():
    # Print start time
    print("Start Time:", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    train_X, test_X, train_y, test_y, timestmps = json_pull()
    train_X = normalize(train_X)
    test_X = normalize(test_X)
    train_y = train_X
    test_y = test_X
    num_testmeas = test_X.shape[0]
    # sess = tf.InteractiveSession()

    # Input and output layers' sizes
    x_size = train_X.shape[1]   # Number of input nodes
    y_size = train_y.shape[1]   # Number of outcomes

    # Symbols
    # These aren't specific values yet. The X and y create input and output nodes to feed the training data through
    X = tf.placeholder(tf.float32, shape=[None, x_size])
    y = tf.placeholder(tf.float32, shape=[None, y_size])

    # Weight initializations
    # Initialize weights randomly. Weights go in between layers (input layer, output layer)
    layer_size = [x_size] + h_size + [y_size]
    ys = [0. for row in range(len(h_size))]         # num outputs equal total num hidden layers
    wts = [0. for row in range(len(h_size) + 1)]    # num wt columns equals num hidden layers + 1
    for i in range(len(h_size) + 1):
        wts[i] = init_weights((layer_size[i], layer_size[i + 1]))
        if i == 0:
            ys[0] = forwardprop(X, wts[0])
        elif i == len(ys):
            yhat = tf.matmul(ys[i - 1], wts[i])
        else:
            ys[i] = forwardprop(ys[i - 1], wts[i])

    # Prediction is on the FINAL layer output
    predict = yhat

    # Setup Backward propagation
    cost    = tf.reduce_mean(np.square(y-yhat))
    # cost = tf.reduce_max(abs(y - yhat))

    # # Set up training using Gradient Descent Optimizer
    # updates = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    # Alternative setup using ADAM Optimizer
    updates = tf.train.AdamOptimizer(0.001).minimize(cost)

    # Run SGD: Stochastic Gradient Descent
    # Begin the C tensorflow session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(num_epochs):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        output_train = sess.run(predict, feed_dict={X: train_X, y: train_y})
        output_test = sess.run(predict, feed_dict={X: test_X, y: test_y})
        train_accuracy = np.ndarray.mean(1. - abs(output_train - train_X))


        if (epoch+1)%100==0:
            print("Epoch = %d, train accuracy = %.2f%%"
                  % (epoch + 1, 100. * train_accuracy))

    # Save the weights if you would like into a csv file. Need to write an eval function
    # weights0 = wts[0].eval()
    # weights1 = wts[1].eval()
    # weights2 = wts[2].eval()
    # weights3 = wts[3].eval()
    # np.savetxt('weights0.csv', weights0, fmt='%.18e', delimiter=',')
    # np.savetxt('weights1.csv', weights1, fmt='%.18e', delimiter=',')
    # np.savetxt('weights2.csv', weights2, fmt='%.18e', delimiter=',')
    # np.savetxt('weights3.csv', weights3, fmt='%.18e', delimiter=',')
    sess.close()

    # Print Ending time
    print("End Time:", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    # Print max and min
    output_test = np.square(output_test - test_X)
    output_avg = np.zeros((num_testmeas, 1))
    for i in range(num_testmeas):
        output_avg[i, 0] = np.ndarray.mean(output_test[i, :])
    maxval = np.argmax(output_avg)
    minval = np.argmin(output_avg)

    print("max index:", maxval)
    print("max index time:", timestmps[maxval])
    print(output_test[maxval, :])
    print("min index:", minval)
    print("min index time:", timestmps[minval])
    print(output_test[minval, :])

    # Plotting the "health scores" of each of the test points
    t = np.arange(1, num_testmeas + 1, 1)
    output_avg = np.squeeze(np.asarray(output_avg))  # Convert to an array
    plt.plot(t, output_avg, 'bo')
    plt.show()

    # # Also plot RMS to see how that differs
    # output_rms = np.squeeze(np.asarray(test_X_orig[0:num_testmeas, 1]))     # Convert to an array

if __name__ == '__main__':
    main()
