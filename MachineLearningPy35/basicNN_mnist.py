import tensorflow as tf

# ------------------------------------------- DATA IMPORT SECTION ---------------------------------------------------- #
# Import mnist data: Requires internet connection to collect
from tensorflow.examples.tutorials.mnist import input_data

# Formatting data so the classification is more machine readable. Input into a folder called "tmp/data"
mnist = input_data.read_data_sets('tmp/data/', one_hot=True)


# --------------------------------------- PARAMETER TWEAKING SECTION ------------------------------------------------- #
# Can tweak how many nodes in each hidden layer, starting out with 500 for now
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10          # There are 10 numbers, 0-9, and we assume each one is in one of those categories
batch_size = 100        # Go through batches of 100 images manipulate the weights

# This is your input data, 28x28 pixels. (image is flattened, can be just string of values).
# All data must be in this shape
x = tf.placeholder('float', [None, 784])        # 28*28=764, this can be dynamically defined
y = tf.placeholder('float')                     # This is your output, what number isi t?

def neural_network_model(data):
    # Create the variables for our layers

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}   # Start with a random set of weights, 784 x n_nodes

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    # For layer 1, do matrix multiplication and add biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']),hidden_1_layer['biases'])
    # Now it is passed through an activation function
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x, y):

    # Run the previous function's model
    prediction = neural_network_model(x)

    # Define optimizer and loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))  # Measures probability error
    learning_rate = 0.001       # This is default
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Cycles feed forwards and backprops
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training the network is all in the below for loops
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):       # just means you don't care variable
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)                   # train next batch
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})    # Passing them x's and y's
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        # After optimization is complete, we compare with the actual label
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


# ---------------------------------------- RUN TRAINING AND ACCURACY ------------------------------------------------- #
train_neural_network(x, y)
