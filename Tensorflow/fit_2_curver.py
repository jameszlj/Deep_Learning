import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    """neural network layers"""
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# make up some fake data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data)-0.05+noise

# detemine inputs dtype
x_s = tf.placeholder(tf.float32, [None, 1])
y_s = tf.placeholder(tf.float32, [None, 1])

# add layer
l_1 = add_layer(x_s, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l_1, 10, 1, activation_function=None)

# computer the cost
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_s - prediction),
                                    reduction_indices=[1]))
# computer the gradient
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# initialize variable(important)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# plot the picture
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

#  learning
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={x_s: x_data, y_s: y_data})
    if i % 50 == 0:
        # to see step improvement
        print(sess.run(loss, feed_dict={x_s: x_data, y_s: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception as ret:
            print(ret)
        predicted_value = sess.run(prediction, feed_dict={x_s:x_data})

        # plot the prediction
        lines = ax.plot(x_data, predicted_value, "r-", lw=5)
        plt.pause(1)






