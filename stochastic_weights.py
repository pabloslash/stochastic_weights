############################################
# Pablo Tostado Marcos
#
#
# Last modified 02/22/2018
#
#
# MNIST_dir = /Users/pablo_tostado/Pablo_Tostado/ML_Datasets/mnist'
#############################################

from __future__ import print_function
from mnist import MNIST
import numpy as np
import pylab as plt
from my_helper import *
import math
import random
from sigmoid_layer import *
from softmax_layer import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
import copy
import time
import IPython as ip

class NeuralNetwork(object):
    def __init__(self, mnist_directory, lr0=None, lr_dampener=None,
                 minibatch_size=128, num_classes=10, magic_sigma=False, alpha=None,
                 log_rate=10):
        self.mnist_directory = mnist_directory
        self.lr_dampener = lr_dampener
        self.holdout_data = None
        self.holdout_data = None
        self.holdout_labels = None
        self.target = None
        self.num_classes = num_classes
        # self.load_data(self.mnist_directory)
        #
        # if lr0 == None:
        #     self.lr0 = 150.0 / self.train_data.shape[0]
        # else:
        #     self.lr0 = lr0
        # self.minibatch_index = 0
        # self.minibatch_size = minibatch_size
        #
        # self.min_loss_holdout = np.inf
        # self.min_loss_weights = None
        #
        self.magic_sigma = magic_sigma
        self.alpha = alpha
        self.log_rate = log_rate
        self.new_epoch = False
        self.epoch_num = 1

    def load_data(self, num_train=0, num_test=0):
        '''
        Will load MNIST data and return the first and last specified number of training & testing images respectively
        Loads all by default: 60k training images, 10k testing images. LOADS FROM THE END.
        '''
        print('Laoding Data...')
        mndata = MNIST(self.mnist_directory)
        train_dat, train_lab = mndata.load_training()
        test_dat, test_lab = mndata.load_testing()
        self.train_data = np.array(train_dat[-num_train:])
        self.train_labels = np.array(train_lab[-num_train:])
        self.test_data = np.array(test_dat[-num_test:])
        self.test_labels = np.array(test_lab[-num_test:])

        self.num_categories = len(list(set(self.train_labels)))
        self.possible_categories = list(set(self.train_labels))
        self.possible_categories.sort()
        print('Data Loaded...')
        # return np.array(train_dat[-num_train:]), np.array(train_lab[-num_train:]), \
        #        np.array(test_dat[-num_test:]), np.array(test_lab[-num_test:])

    def z_score_data(self):
        '''
        Images /127.5 - 1 so that they are in range [-1,1]
        '''
        self.train_data = self.train_data/127.5 -1
        self.test_data = self.test_data/127.5 - 1
        # return self.train_data, self.test_data

    def fan_in(self, inputs):
        '''
        Calculates 1/sqrt(inputs) to unit to initialize the weights of the network
        '''
        return 1/(inputs**(1/2.0))

    def initialize_weights(self, num_hidden_units, mu=0):
        '''
        Right now only configurable for 1 hidden layer. Specify number of units
        '''
        num_input_units = self.train_data.shape[1]
        self.w_il = np.random.normal(mu, self.fan_in(num_input_units+1), (num_input_units+1, num_hidden_units))
        self.w_ol = np.random.normal(mu, self.fan_in(num_hidden_units+1), (num_hidden_units+1, self.num_classes))

    def prefix_data(self):
        '''
        Images /127.5 - 1 so that they are in range [-1,1]
        '''
        self.train_data = add_bias_term(self.train_data)
        self.test_data = add_bias_term(self.test_data)
        # return self.train_data, self.test_data

    def assign_holdout(self, percent):
        percent /= 100.0
        num_held = int(self.train_data.shape[0] * percent)
        self.train_data = self.train_data[:-num_held]
        self.train_labels = self.train_labels[:-num_held]
        self.holdout_data = self.train_data[-num_held:]
        self.holdout_labels = self.train_labels[-num_held:]
        print('Assigned holdout data')

    def rand_minibatch(self, batch_size):
        '''
        randomizes order of samples
        '''
        index_rand = np.array(random.sample(xrange(len(self.train_data)), batch_size))
        rand_batch = np.array([self.train_data[n] for n in index_rand])
        rand_labels = np.array([self.train_labels[n] for n in index_rand])
        return rand_batch,rand_labels

    def forward_ih(self, input_batch, derivative=False):
        '''
        The forward propagation between the input layer and the hidden layer with
        either a sigmoid or hyperbolic tangent sigmoid.
        '''
        #input to hidden
        a_j = np.dot(input_batch, self.w_il)   # Weighted sum of inputs
        # print(a_j.shape)

        # z_j = hyperbolic_tangent(a_j, derivative)
        z_j = sigmoid(a_j, derivative)                          # Activation Function
        return z_j

    def get_prediction_error(self, input_i, input_l):
        z_j = self.forward_ih(input_i)
        z_j = add_bias_term(z_j)
        y_k = self.forward_ho(z_j)
        pred_l = np.argmax(y_k, 1)   #Predicted labels
        return 100.0 * (np.sum(pred_l == input_l)) / (1.0 * input_l.shape[0]) #Return ACCURACY

    def loss_funct(self, input_i, input_l):
        '''
        Calculates the cross entropy, the loss function.
        '''
        z_j = self.forward_ih(input_i)
        z_j = add_bias_term(z_j)
        y = self.forward_ho(z_j)
        t = one_hot_encoding(input_l)
        #Normalize w.r.t # training examples and #categories
        return (-1.0 / (input_i.shape[0] * w_ol.shape[1])) * (np.sum(t * np.log(y)))


    def forward_ho(self, hidden_activations):
        '''
        The forward propagation between the hidden layer and the output layer with
        softmax applied to the output layer.
        '''
        #hidden to output
        a_k = np.dot(hidden_activations, self.w_ol) #+ bias_o  # Weighted sum of inputs
        y_k = softmax(a_k)                                              # Activation Function
        return y_k

    def get_dk_gradient(self, z_j, l):
        y = self.forward_ho(z_j)              # Recalculate classification probs
        t = one_hot_encoding(l)                # One-hot encode labels
        return self.delta_k(y,t)

    def delta_k(self, y, t):
        '''
        Delta_K for output units
        '''
        return (t - y)

    def loss_funct(self, input_i, input_l):
        '''
        Calculates the cross entropy, the loss function.
        '''
        z_j = self.forward_ih(input_i)
        z_j = add_bias_term(z_j)
        y = self.forward_ho(z_j)
        t = one_hot_encoding(input_l)
        #Normalize w.r.t # training examples and #categories
        return (-1.0 / (input_i.shape[0] * self.w_ol.shape[1])) * (np.sum(t * np.log(y)))

    def backprop_hi(self, x, d_k, lr):
        '''
        Backpropagation between the hidden layer and the input layer. Function
        returns the update for w_ij.
        '''
        g_h_der = self.forward_ih(x, derivative=True)             # g'(a_j)
        d_j = np.transpose(g_h_der) * (np.dot(self.w_ol[1:,:], np.transpose(d_k)))
        d_Eij = np.transpose( np.dot(d_j, x) )                         # -dEij = d_j * x_i

        w_ih_update = lr * d_Eij
        return w_ih_update

    def backprop_oh(self, z_j, d_k, lr):
        '''
        Backpropagation between the output layer and the hidden layer. Function
        returns the update for w_jk.
        '''
        d_Ejk = np.dot(np.transpose(z_j), d_k)
        w_jk_update = lr * d_Ejk                # Update weights
        return w_jk_update

    def plot_loss_acc(self):

        plt.figure()
        plt.plot(self.Ltr,  label='Training data' )
        plt.plot(self.Lval,  label='Validation data')
        plt.plot(self.Ltr,  label='Testing data')
        plt.title('Cross Entropy loss function')
        plt.xlabel('# Epochs')
        plt.ylabel('Cross Entropy')
        plt.legend(loc='upper right')
        plt.show(block=False)

        plt.figure()
        plt.plot(self.Etr,  label='Training data')
        plt.plot(self.Eval,  label='Validation data')
        plt.plot(self.Etest,  label='Testing data (%.2f%s)' %(np.max(self.Etest), '%'))
        plt.title('Prediction Accuracy')
        plt.xlabel('# Epochs')
        plt.ylabel('% accuracy')
        plt.legend(loc='lower right')
        plt.show(block=False)

    def train(self, epochs, lr, batch_size = 128, anneal=True, log_rate=None,
                         l1=False, l2=False, lamb=None):

        self.Ltr, self.Lval, self.Ltest = [], [], []
        self.Etr, self.Eval, self.Etest = [], [], []

        iterations = self.train_data.shape[0] / batch_size #Go through whole train_set
        for ep in xrange(epochs):
            for it in xrange(iterations):

                batch_i, batch_l = self.rand_minibatch(batch_size)

                #FORWARD PROP
                z_j = self.forward_ih(batch_i) # Activation Function of hidden units
                z_j = add_bias_term(z_j) # Add extra 1st column to hidden activations for biases
                #For_Prop: hidden to output
                y_k = self.forward_ho(z_j) # Activation Function of output units

                # BACKPROP:
                d_k = self.get_dk_gradient(z_j, batch_l)
                # 1st: hidden to input (Bc we need old w_jk)
                w_il_update = self.backprop_hi(batch_i, d_k, lr) # Update w_ij weights
                self.w_il += w_il_update

                # 2nd: output to hidden
                w_ol_update = self.backprop_oh(z_j, d_k, lr) # Update w_jk weights
                self.w_ol += w_ol_update

            self.Ltr.append(self.loss_funct(self.train_data, self.train_labels))
            self.Lval.append(self.loss_funct(self.holdout_data, self.holdout_labels))
            self.Ltest.append(self.loss_funct(self.test_data, self.test_labels))

            self.Etr.append(self.get_prediction_error(self.train_data, self.train_labels))
            self.Eval.append(self.get_prediction_error(self.holdout_data, self.holdout_labels))
            self.Etest.append(self.get_prediction_error(self.test_data, self.test_labels))


            print ('Epoch ' + str(ep) + '| Loss = '+ str(self.loss_funct(self.test_data, self.test_labels)) )

        self.plot_loss_acc()
