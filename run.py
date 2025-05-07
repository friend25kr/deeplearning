import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
list_training_data = list(training_data)
list_test_data = list(test_data)

net = network.Network([784, 30, 10])
net.SGD(list_training_data, 30, 10, 3.0, test_data=list_test_data)