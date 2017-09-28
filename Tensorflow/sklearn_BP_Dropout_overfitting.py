import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    """add neural network layer"""
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    #  dropout main function
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + "/outputs", outputs)
    return outputs

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)  # dropout
x_s = tf.placeholder(tf.float32, [None, 64])  # 8x8
y_s = tf.placeholder(tf.float32, [None, 10])

# add output layer
l_1 = add_layer(x_s, 64, 50, "l_1", activation_function=tf.nn.tanh)
prediction = add_layer(l_1, 50, 10, "l_2", activation_function=tf.nn.softmax)

# the error between prediction and real_data
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(
        y_s * tf.log(prediction), reduction_indices=[1]
    )
)
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()

# merge all training picture
merged = tf.summary.merge_all()

# above pictures save as "logs/"
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)

# initialize
sess.run(tf.global_variables_initializer())

# training
for i in range(500):
    sess.run(train_step, feed_dict={x_s: X_train, y_s: y_train, keep_prob: 0.5})
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={x_s: X_train, y_s: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={x_s: X_test, y_s: y_test, keep_prob: 1})

        # loading to summary
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)



