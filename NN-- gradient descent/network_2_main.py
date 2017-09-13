import network_2
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network_2.Network([784, 30, 10], cost=network_2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
# Epoch 29 training complete
# Accuracy on evaluation data: 9539 / 10000

net = network_2.Network([784, 30, 10], cost=network_2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.5, 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)

# Epoch 29 training complete
# Cost on training data: 0.13504619359815603
# Accuracy on training data: 48502 / 50000
# Cost on evaluation data: 0.6752309679907801
# Accuracy on evaluation data: 9631 / 10000
