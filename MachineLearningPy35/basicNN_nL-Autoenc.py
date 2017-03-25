"""
Iris dataset has three classification classes: setosa(1), virginica(2), versicolor(3)
Set has 150 samples, 4 features each
Use a simple feedforward neural network with a single hidden layer to predict the class
4 input nodes ---> hidden layer ---> hidden layer ---> .... ---> 3 output nodes
"""
import tensorflow as tf
import numpy as np
import json
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Enter number of nodes in each fully connected layer
# h_size is a list that holds the number of nodes in each layer
h_size = [64, 4, 64]
num_epochs = 10000
RANDOM_SEED = 55
tf.set_random_seed(RANDOM_SEED)

jfile = open("C:\\Users\\Simon\\Documents\\Data\\measurement_data_40_L3_z.json", "r")
json_dict = json.loads(jfile.read())
############################################################################
# Change the following function to get the data that you are using

def json_pull():
    """ Pull data from json file into a numpy matrix"""
    """
    train_mat will be the input matrix which we would like to add features to
    featrs will be a tuple of features we would like to add to the training matrix.
    """
    num_meas = 1000  # len(json_dict['measurements'])
    num_feat = 8

    # Define the training set
    trainmatx = np.matrix([[0.0 for col in range(num_feat)] for row in range(num_meas)])
    print(trainmatx.shape)
    for i in range(num_meas):
        trainmatx[i, 0] = json_dict['measurements'][i]['data']['z']['time_domain_features']['p2p']
        trainmatx[i, 1] = json_dict['measurements'][i]['data']['z']['time_domain_features']['rms']
        trainmatx[i, 2] = json_dict['measurements'][i]['data']['z']['time_domain_features']['peak']
        trainmatx[i, 3] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_1x']
        trainmatx[i, 4] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_2x']
        trainmatx[i, 5] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_3x']
        trainmatx[i, 6] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_3x']
        trainmatx[i, 7] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_3x_to_10x_sum']

    nanarr = []
    for i in range(trainmatx.shape[0]):
        for j in range(num_feat):
            if math.isnan(trainmatx[i, j]):
                nanarr.append(i)
                break
    trainmatx = np.delete(trainmatx, nanarr, axis=0)

    # Define the test set
    all_X = trainmatx
    all_Y = trainmatx

    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def select_features(feature_str):
    """ 4097 frequency bins in the frequency array """
    num_feat = len(feature_str)

def normalize(train_matrix):
    """ Normalize the training matrix (and also set to unit variance?) """
    train_matrix -= np.mean(train_matrix, axis=0)
    train_matrix /= np.std(train_matrix, axis=0)

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
    train_X, test_X, train_y, test_y = json_pull()
    train_X = normalize(train_X)
    test_X = normalize(test_X)
    train_y = normalize(train_y)
    test_y = normalize(test_y)
    sess = tf.InteractiveSession()

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
    cost    = tf.reduce_mean(abs(y-yhat))

    # Set up training using Gradient Descent Optimizer
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # # Set up training using Gradient Descent Optimizer
    # # Alternative setup using ADAM Optimizer
    # updates = tf.train.AdamOptimizer(0.001).minimize(cost)

    # Run SGD: Stochastic Gradient Descent
    # Begin the C tensorflow session
    # sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(num_epochs):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        output_train = sess.run(predict, feed_dict={X: train_X, y: train_y})
        output_test = sess.run(predict, feed_dict={X: test_X, y: test_y})
        train_accuracy = np.ndarray.mean(1 - abs(output_train - train_X))
        # train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
        #                          sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.ndarray.mean(1 - abs(output_test - test_X))

        # train_accuracy = train_y/sess.run(predict, feed_dict={X: train_X, y: train_y}))
        # test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
        #                     sess.run(predict, feed_dict={X: test_X, y: test_y}))
        if epoch%49==0:
            print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                  % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    # Save the weights if you would like into a csv file. Need to write an eval function
    weights0 = wts[0].eval()
    weights1 = wts[1].eval()
    weights2 = wts[2].eval()
    weights3 = wts[3].eval()
    np.savetxt('weights0.csv', weights0, fmt='%.18e', delimiter=',')
    np.savetxt('weights1.csv', weights1, fmt='%.18e', delimiter=',')
    np.savetxt('weights2.csv', weights2, fmt='%.18e', delimiter=',')
    np.savetxt('weights3.csv', weights3, fmt='%.18e', delimiter=',')
    sess.close()


if __name__ == '__main__':
    main()
