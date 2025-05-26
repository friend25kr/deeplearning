import mnist_loader
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
list_training_data = list(training_data)
list_test_data = list(test_data)

print(list_test_data[500][0], list_test_data[500][1])

net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(list_training_data[:1000], 400, 10, 0.5, evaluation_data=list_test_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)