import tensorflow as tf
import numpy as np


# getcwd gets current working directory
# cwd = os.getcwd()
# mnist = input_data.read_data_sets(cwd + "/tmp/data/", one_hot=True)
# X = tf.placeholder(tf.float32, shape=[None, 100])
# y = tf.placeholder(tf.float32, shape=[None, 100])

# ----------------------------------- Understanding how to make a tensorflow matrix ----------------------------------
x = [[[1, 2, 3],
      [4, 5, 6]],
     [[7, 8, 9],
      [10, 11, 12]]]
y = [[1, 2, 3],
     [4, 5, 6]]

w = [[1, 2, 3]]

print(w[0][2])

# permute transpose means you have to indicate which dimension to tranpose in what order
z = tf.transpose(x, perm=[2,0,1])

# Have to run session in order to view the variable values. Nothing is done without the session
sess = tf.Session()
print(sess.run(z))
# ----------------------------------- Understanding how to make a tensorflow matrix ----------------------------------