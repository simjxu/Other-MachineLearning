from __future__ import print_function
import tensorflow as tf

# Import mnist data: Requires internet connection to collect
from tensorflow.examples.tutorials.mnist import input_data

# Formatting data so the classification is more machine readable
mnist = input_data.read_data_sets('tmp/data/', one_hot=True)

# Set hyperparameters
learning_rate = 0.001
training_iters = 200000         # how many iterations
batch_size = 128                # we have 128 samples
display_step = 10               # every 10 iterations we display what's happening

# Set network parameters
# 28 x 28 image
n_input = 784
n_classes = 10                  # Number of possible classification cclasses
dropout = 0.75                  # Randomly turns off some neurons to allow for a more generalized model (not overfit) REVISIT

x = tf.placeholder(tf.float32, [None, n_input])     # What does "None" mean again? This means dimension can be of any length
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# 2d convolutional layer
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides])
    # x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Pooling layer
def maxpool2d(x, k=2):      # Subsamples of the convolutional layer
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

# Create the model
def conv_net(x, weights, biases, dropout):
    # Reshape input
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Create the convolutional layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

    # Max pooling
    conv1 = maxpool2d(conv1, weights['wc2'], biases['bc2'])

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2,k=2)

    # Have two layers now, so we need to create a fully connected layer. Fully connected is generic and is connected to
    # everything in the previuous layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].getshape().as_list()])  # Needs to be reshaped
    fc1 = tf.add(tf.matmul(fc1, weights['wd1'], biases['bc1']))
    fc1 = tf.nn.relu(fc1)           # Apply RELU activation function
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output will predict our class
    out = tf.add(tf.matmul(fc1, weights['out'], biases['out']))
    return out

# Create weights
weights = {
    'wc1': tf.Variable(tf.random_normal([5,5,1,32])),           # 5x5 convolution with one input and 32 outputs (bits
    'wc2': tf.Variable(tf.random_normal([5,5,32,64])),          # 32 different connections. Splitting the picture even more
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),      # fully connected layer, 7x7x64 inputs 1024 outputs
    'out': tf.Variable(tf.random_normal(1024, n_classes))
}

# Construct Model
pred = conv_net(x, weights, biases, keep_prob)

# Define optimizer and loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))     # Measures probability error
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate Model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the Variables
init = tf.initialize_all_variables()

# Launch Computation graph (neural net) Graph encapsulated by a session
# with tf.Session as sess:
#     sess.run(init)
#     step = 1
#     # keep training until you reach max interations
#     while step * batch_size < training_iters:
#         sess.run(optimizer,feed_dict{x: batch_x: y: batch_y: })
#
#         print('iteration step')
