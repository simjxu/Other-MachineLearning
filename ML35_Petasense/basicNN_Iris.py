"""
Iris dataset has three classification classes: setosa(1), virginica(2), versicolor(3)
Set has 50 samples, 4 features each
Use a simple neural network with a single hidden layer to predict the class
4 input nodes ---> hidden layer ---> 3 output nodes
"""
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    # The sigma function
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))

    # # Alternative function using ReLU
    # h    = tf.nn.relu(tf.matmul(X, w_1))

    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def main():
    # Get the Iris data
    # train_X: 100 x 5 matrix, first column all ones (bias), then 4 features
    # test_X: 50 x 5 matrix, first column all ones (bias), then 4 features
    # train_y: 100 x 3 matrix, one hot for classification of each of the training samples
    # test_y: 50 x 3 matrix, one hot for classification of each of the test samples
    train_X, test_X, train_y, test_y = get_iris_data()
    sess = tf.InteractiveSession()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 256                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    # These aren't specific values yet. The X and y create input and output nodes to feed the training data through
    X = tf.placeholder(tf.float32, shape=[None, x_size])
    y = tf.placeholder(tf.float32, shape=[None, y_size])

    # Weight initializations
    # Initialize weights randomly
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # # Setup Forward propagation with a (additional) hidden layer
    # y_1     = forwardprop(X, w_1, w_2)

    # Setup Forward propagation for FINAL hidden layer
    # Prop with a sigmoid function or ReLU
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Setup Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))

    # Set up training using Gradient Descent Optimizer
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # # Set up training using Gradient Descent Optimizer
    # # Alternative setup using ADAM Optimizer
    # updates = tf.train.AdamOptimizer(0.001).minimize(cost)

    # Run SGD: Stochastic Gradient Descent
    # Begin the C++ tensorflow session
    # sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(10):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    # # Save the weights if you would like into a csv file. Need to write an eval function
    # weights1 = w_1.eval()
    # np.savetxt('weights1.csv', weights1, fmt='%.18e', delimiter=',')
    sess.close()


if __name__ == '__main__':
    main()
