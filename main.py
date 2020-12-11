import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages, un-comment if want to see but they annoy me

import matplotlib.pyplot as plt
import numpy as np
from time import time

from tensorflow.keras.utils import to_categorical #use for one-hot encoding labels
from tensorflow.keras.optimizers import SGD #optimizer

from resnet import ResNet
from bayesian_resnet import BayesianResNet
from utils import loadData, calculate_accuracy

def get_models(load_saved_model = False):
    ''''''
    if load_saved_model:
        resnet = ResNet(load = True) #load resnet
        bayesian_resnet = BayesianResNet(input_shape = (256, 256, 1), num_classes = 2, load = True) #load bayesian resnet
    else:
        resnet = ResNet(input_shape = (256, 256, 1), num_classes = 2) #init resnet
        bayesian_resnet = BayesianResNet(input_shape = (256, 256, 1), num_classes = 2) #init bayesian resnet
    return resnet, bayesian_resnet

def train_models(resnet, bayesian_resnet, num_epochs = 10):
    ''''''
    #load training and validation datasets
    train, train_labels = loadData('data/ros_data.npz')
    val, val_labels = loadData('data/val.npz')
    
    opt = SGD(learning_rate = 1e-3) #setup optimizer
    
    #resnet.fit(np.expand_dims(train, axis = -1), to_categorical(train_labels),
    #          validation_data = [np.expand_dims(val, axis = -1), to_categorical(val_labels)],
    #           epochs = num_epochs, batch_size = 5, optimizer = opt, save = True) #train model
    
    bayesian_resnet.fit(np.expand_dims(train, axis = -1), to_categorical(train_labels),
                        validation_data = [np.expand_dims(val, axis = -1), to_categorical(val_labels)],
                        epochs = num_epochs, batch_size = 5, optimizer = opt, save = True) #train model

def main():
    ''''''

    resnet, bayesian_resnet = get_models(load_saved_model = False) #load or create the models
    train_models(resnet, bayesian_resnet, num_epochs = 10) #train resnet and bayesian resnet until number of epochs trained reaches num_epochs

    #test, test_labels = loadData('data/test.npz');
    
    #start_time = time()
    #pred1 = resnet.predict(np.expand_dims(test, axis = -1))
    #print(pred1.shape, 'Time: ', '{:.2f} seconds'.format(time() - start_time))
    #print(calculate_accuracy(pred1, test_labels)*100)
                   
    #start_time = time()
    #pred2, entropy = bayesian_resnet.predict(np.expand_dims(test, axis = -1), mc_steps = 10)
    #print(pred2.shape, 'Time: ', '{:.2f} seconds'.format(time() - start_time))
    #print(calculate_accuracy(pred2, test_labels)*100)
    
if __name__ == '__main__':
    main()