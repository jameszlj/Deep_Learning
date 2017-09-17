import theano.tensor as T
import  numpy as np
import theano
import matplotlib.pyplot as plt


class Layer(object):
    """
    define the layer like this :
    l1 = Layer(inputs, in_sizes, out_size, out_size, activation_function)
    l2 = Layer(l1.outputs, 10, 1, None)
    """
    def __init__(self, inputs, in_sizes, out_sizes, activation_function):
        self.W = theano.shared(np.random.normal(0, 1, (in_sizes, out_sizes)))
        self.b = theano.shared(np.zeros((out_sizes,)) + 0.1)
        self.Wx_plus_b = T.dot(inputs, self.W) + self.b
        self.activation_function = activation_function
        if activation_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)


# make up some fake data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# increase noise
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data)-0.05 + noise # y=x^2 -0.05+noise
# show the fake data
# plt.scatter(x_data, y_data)
# plt.show()

# detemine th inputs dtype
x = T.dmatrix("x")
y = T.dmatrix("y")

# add layers
l_1 = Layer(x, 1, 10, T.nnet.relu)
l_2 = Layer(l_1.outputs, 10, 1, None)

# computer the cost
cost = T.mean(T.square(l_2.outputs - y))

# computer the gradient
g_W1, g_b1, g_W2, g_b2 = T.grad(cost, [l_1.W, l_1.b, l_2.W, l_2.b])

# applay gradient descent
learning_rate = 0.05
train = theano.function(
    inputs=[x, y],
    outputs=[cost],
    updates=[(l_1.W, l_1.W - learning_rate * g_W1),
             (l_1.b, l_1.b - learning_rate * g_b1),
             (l_2.W, l_2.W - learning_rate * g_W2),
             (l_2.b, l_2.b - learning_rate * g_b2)])

# prediction
predict = theano.function(inputs=[x], outputs=l_2.outputs)

# plot the fake data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()  # open the two picture
plt.show()

for i in range(1000):
    # training
    err = train(x_data, y_data)
    if i % 50 == 0:
        # visualization
        try:
            ax.lines.remove(lines[0])
        except Exception as ret:
            print(ret)
        prediction_value = predict(x_data)
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(1)




