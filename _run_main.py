#Main to run neural network
from stochastic_weights import *
from my_helper import *


# Initialize variables:
directory = '/Users/pablo_tostado/Pablo_Tostado/ML_Datasets/mnist'
hu = 200            #Num hidden units
hold_out_per = 10   #Holdout percentage
epochs = 40         #Each epoch consists of a run trhough all training data
lr = 0.01           #Learning rate


# Run Neural Net
nn = NeuralNetwork(directory, magic_sigma=True)
nn.load_data()
nn.z_score_data()
nn.initialize_weights(hu) # Num hidden units in hidden layer
nn.prefix_data()
nn.assign_holdout(hold_out_per)

nn.train(epochs, lr) # Itertions + Learning rate
