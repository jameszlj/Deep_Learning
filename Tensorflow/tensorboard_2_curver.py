import numpy as np
import tensorflow as tf


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    """add neural network layer"""
    layer_name = "layer%s" % n_layer
    with tf.name_scope(layer_name):
        # define weights name
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
            tf.summary.histogram(layer_name+'/weights', Weights)
        # define biase
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name+'/biases', biases)
        # define z
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            out_puts = Wx_plus_b
        else:
            out_puts = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs', out_puts)
        return out_puts


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.05 + noise

with tf.name_scope("inputs"):
    # define placeholder for inputs to network
    x_s = tf.placeholder(tf.float32, [None, 1], name='x_in')
    y_s = tf.placeholder(tf.float32, [None, 1], name='y_in')

# add hidden layer
l_1 = add_layer(x_s, 1, 10, n_layer=1, activation_function=tf.nn.relu)
prediction = add_layer(l_1, 10, 1, n_layer=2, activation_function=None)

# the error between prediction and real data
with tf.name_scope("loss"):
    loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.square(y_s - prediction), reduction_indices=[1]
        )
    )
    tf.summary.scalar("loss", loss)

# training data
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

# merge all training picture
merged = tf.summary.merge_all()

#  above pictures saves as "logs/"
writer = tf.summary.FileWriter("logs/", sess.graph)

# initialize
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_step, feed_dict={x_s: x_data, y_s: y_data})
    if i % 50 == 0:
        res = sess.run(merged, feed_dict={x_s: x_data, y_s: y_data})
        writer.add_summary(res, i)

