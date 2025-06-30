"""network3.py
~~~~~~~~~~~~~~

A pytensor-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on pytensor, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the pytensor documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

Written for pytensor 0.6 and 0.7, needs some changes for more recent
versions of pytensor.

"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import pytensor
import pytensor.tensor as T # Theano has been renamed to PyTensor
import pytensor.tensor as T
from pytensor.tensor.special import softmax
from pytensor.tensor.conv import conv2d
from pytensor.sparse.sandbox import sp as downsample
from pytensor.tensor.random.utils import RandomStream

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from pytensor.tensor.math import sigmoid
from pytensor.tensor import tanh


#### Constants
GPU = True
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then modify network3.py\nto set the GPU flag to False.")
    try: pytensor.config.device = 'gpu'
    except: pass # it's already set
    pytensor.config.floatX = 'float32'
else:
    print("Running with a CPU.  If this is not desired, then the modify network3.py to set\nthe GPU flag to True.")

#### Load the MNIST data
def load_data_shared(filename="/content/drive/MyDrive/Colab Notebooks/mnist_expanded.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    def shared(data):
        """Place the data into shared variables.  This allows pytensor to copy
        the data to the GPU, if one is available.

        """
        # Assuming data is a tuple/list like (features, labels)
        features = data[0]
        labels = data[1]

        # Convert features to a NumPy array, ensuring consistent shape if necessary
        # This part might need adjustment based on the actual structure of your data
        shared_x = pytensor.shared(
            np.asarray(features, dtype=pytensor.config.floatX), borrow=True)

        # Convert labels to a NumPy array
        shared_y = pytensor.shared(
            np.asarray(labels, dtype=pytensor.config.floatX), borrow=True) # Keep as floatX for now, cast later

        return shared_x, T.cast(shared_y, "int32")

    # Apply the shared function to each dataset
    shared_training_data = shared(training_data)
    shared_validation_data = shared(validation_data)
    shared_test_data = shared(test_data)

    return [shared_training_data, shared_validation_data, shared_test_data]

#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = pytensor.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = pytensor.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = pytensor.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = pytensor.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(int(num_training_batches)):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print(f"Training mini-batch number {iteration}")
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(int(num_validation_batches))])
                    print(f"Epoch {epoch}: validation accuracy {validation_accuracy}")
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in range(int(num_test_batches))])
                            print(f"The corresponding test accuracy is {test_accuracy}")
        print("Finished training network.")
        print(f"Best validation accuracy of {best_validation_accuracy} obtained at iteration {best_iteration}")
        print(f"Corresponding test accuracy of {test_accuracy}")

#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = pytensor.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=pytensor.config.floatX),
            borrow=True)
        self.b = pytensor.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=pytensor.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv2d(
            input=self.inpt, filters=self.w,
            input_shape=self.image_shape, filter_shape=self.filter_shape)

        conv_out_shape = conv_out.shape
        pooled_h = conv_out_shape[2] // self.poolsize[0]
        pooled_w = conv_out_shape[3] // self.poolsize[1]
        reshaped_for_pooling = conv_out.reshape((
            conv_out_shape[0], conv_out_shape[1],
            pooled_h, self.poolsize[0],
            pooled_w, self.poolsize[1]))
        pooled_out = T.max(reshaped_for_pooling, axis=(3, 5))

#        pooled_out_raw = downsample.max_pool(images=conv_out, imgshp=(24, 24), maxpoolshp=self.poolsize)
#        pooled_out = T.as_tensor_variable(pooled_out_raw)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = self.output.flatten(2)
        self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = pytensor.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=pytensor.config.floatX),
            name='w', borrow=True)
        self.b = pytensor.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=pytensor.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = pytensor.shared(
            np.zeros((n_in, n_out), dtype=pytensor.config.floatX),
            name='w', borrow=True)
        self.b = pytensor.shared(
            np.zeros((n_out,), dtype=pytensor.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    rng = np.random.RandomState(0)
    srng = RandomStream(rng.randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, pytensor.config.floatX)

training_data, validation_data, test_data = load_data_shared()
mini_batch_size = 10

#net = Network([FullyConnectedLayer(n_in=784, n_out=100), SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
#net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

#net = Network([
#        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
#                      filter_shape=(20, 1, 5, 5),
#                      poolsize=(2, 2)),
#        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
#                      filter_shape=(40, 20, 5, 5),
#                      poolsize=(2, 2)),
#        FullyConnectedLayer(n_in=40*4*4, n_out=100),
#        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

#net.SGD(training_data, 60, mini_batch_size, 0.1,
#            validation_data, test_data)ex

#net = Network([
#        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
#                      filter_shape=(20, 1, 5, 5),
#                      poolsize=(2, 2),
#                      activation_fn=ReLU),
#        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
#                      filter_shape=(40, 20, 5, 5),
#                      poolsize=(2, 2),
#                      activation_fn=ReLU),
#        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
#        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

#net.SGD(training_data, 60, mini_batch_size, 0.03,
#            validation_data, test_data, lmbda=0.1)

net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(
            n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        FullyConnectedLayer(
            n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
        mini_batch_size)
net.SGD(training_data, 40, mini_batch_size, 0.03,
            validation_data, test_data)