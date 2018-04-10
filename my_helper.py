from __future__ import print_function
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import IPython as ip


#Prints Image. Input: 1D array 784
def mnist_printer(mnist_array, save=False):
    pixmap = weights_to_2d(mnist_array).astype(float)
    # print pixmap.shape #28x28
    plt.figure
    plt.imshow(pixmap, cmap=cm.gray, interpolation='nearest')
    plt.show(block=False)

# turns 1D array (784 weights) into 2D array of to 28x28
def weights_to_2d(weights):
    dim1 = int(np.sqrt(len(weights)))
    dim2 = int(len(weights) / dim1)
    weights = weights[:dim1*dim2] # This is for adding the occlusions.
    return copy.deepcopy(np.reshape(weights, (dim1, dim2)))

def sigmoid(x, derivative=False):
    y = 1 / (1 + np.exp(-1 * x))
    if (derivative == True):
        return y*(1-y)
    else:
        return y

def dl1(self, w):
    return np.sign(w)

def dl2(self, w):
    return 2 * w

def add_bias_term(x_array):
    '''
    Add a 1 in front of every input vector that accounts for the bias weight
    '''
    a = np.array(x_array)
    x_bias = np.insert(a, 0, 1, axis=1)
    return np.array([np.append(1,x) for x in x_array])

def one_hot_encoding(label_to_1hotencode):
    '''
    Makes labels into vectors
    '''
    encoded_list = list()
    for label in label_to_1hotencode:
        label_zero = [0 for i in xrange(10)]
        label_zero[label] = 1
        encoded_list.append(label_zero)
    return np.array(encoded_list)

def softmax(activation_k):
    exp_ak = np.exp(activation_k)  # Exp of my class
    sum_exp_ak = np.sum(exp_ak, 1) # Sum of exp of classes
    sum_exp_ak = np.reshape(sum_exp_ak, (exp_ak.shape[0], 1))
    sum_exp_ak = np.repeat(sum_exp_ak, exp_ak.shape[1], axis=1)
    return exp_ak / (1.0 * sum_exp_ak) # Normalized outputs of classifier
