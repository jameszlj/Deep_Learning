import mnist_loader
import network_1
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print('训练数据')
print(type(training_data))
print(len(training_data))
print(training_data[0][0].shape)
print(training_data[0][1].shape)

print('验证数据')
print(len(validation_data))

print('测试数据')
print(len(test_data))

# net = network_1.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)  # Epoch 29: 9463 / 10000

# net = network_1.Network([784, 100, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)  # Epoch 29: 9631 / 10000 较之前运行耗时多但准确率提高

net = network_1.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 0.01, test_data=test_data)  # Epoch 29: 7621 / 10000
