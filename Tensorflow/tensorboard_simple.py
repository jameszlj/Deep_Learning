import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_function=None):
    """load neural network layer"""
    with tf.name_scope("layer"):
        # define weights name
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        # define biase
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        # define z
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


with tf.name_scope("inputs"):
    # define placeholder for inputs to network
    x_s = tf.placeholder(tf.float32, [None, 1], name="x_in")
    y_s = tf.placeholder(tf.float32, [None, 1], name="y_in")

# add  hidden layer
l_1 = add_layer(x_s, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l_1, 10, 1, activation_function=None)

# the error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.square(y_s - prediction), reduction_indices=[1]
        )
    )

# training
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

# the picture save as "logs/"by following
writer = tf.summary.FileWriter("logs/", sess.graph)

# initialize
sess.run(tf.global_variables_initializer())