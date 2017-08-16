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
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# train
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# create session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# TRAINING
for _ in range(1000):
    # 'stochastic gradient descent' training
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

