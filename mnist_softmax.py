import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# None means dimension of any length.
# 784 means 784 dimension
x = tf.placeholder(tf.float32, [None, 784])

# initialize full of zeros
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# predicted
# ex) tf.matmul(x, W) + b => tf.nn.softmax_cross_entropy_with_logits
# ex) because this formulation is numerically unstable
y = tf.nn.softmax(tf.matmul(x, W) + b)

# true
y_ = tf.placeholder(tf.float32, [None, 10])

# mean is like average
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indicise=[1]))

