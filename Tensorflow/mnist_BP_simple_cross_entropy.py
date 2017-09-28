import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None):
    """add the neural network"""
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    baises = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + baises
    if activation_function is None:
        out_puts = Wx_plus_b
    else:
        out_puts = activation_function(Wx_plus_b)
    return out_puts


def compute_accuracy(v_xs, v_ys):
    """compute distinguish picture's accuracy"""
    global prediction
    y_pre = sess.run(prediction, feed_dict={x_s: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    print(correct_prediction)
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32)
    )
    print(accuracy)
    result = sess.run(accuracy, feed_dict={x_s: v_xs, y_s: v_ys})
    return result

# define placeholder for inputs to network
x_s = tf.placeholder(tf.float32, [None, 784])  # 28x28
y_s = tf.placeholder(tf.float32, [None, 10])

# add outputlayer
prediction = add_layer(x_s, 784, 10, activation_function=tf.nn.softmax)

# the error between prediction and real data(entropy Gain)
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(
        y_s * tf.log(prediction), reduction_indices=[1]
    )
)
# training
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.Session()

# initialize
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x_s:batch_xs, y_s:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
# result
# Tensor("Equal_19:0", shape=(10000,), dtype=bool)
# Tensor("Mean_20:0", shape=(), dtype=float32)
# 0.8005