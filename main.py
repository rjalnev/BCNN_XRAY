import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages, un-comment if want to see but they annoy me

import matplotlib.pyplot as plt
import numpy as np
from time import time

from tensorflow.keras.utils import to_categorical #use for one-hot encoding labels
from tensorflow.keras.optimizers import Adam #optimizer

from resnet import ResNet
from bayesian_resnet import BayesianResNet
from utils import loadData, calculate_accuracy

def train_models():
    ''''''
    train, train_labels = loadData('data/ros_data.npz')
    val, val_labels = loadData('data/val.npz')
    
    #resnet with 4 stages with [1, 2, 2, 1] number of residual blocks
    resnet = ResNet(input_shape = (256, 256, 1), num_classes = 2) #init model
    opt1 = Adam(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False) #setup optimizer
    resnet.fit(np.expand_dims(train, axis = -1), to_categorical(train_labels),
                validation_data = [np.expand_dims(val, axis = -1), to_categorical(val_labels)],
                epochs = 25, batch_size = 5, optimizer = opt1, save = True) #train model
                
    #bayesian resnet
    bayesian_resnet = BayesianResNet(input_shape = (256, 256, 1), num_classes = 2, kl_weight = 1/train.shape[0]) #init model
    opt2 = Adam(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False) #setup optimizer
    bayesian_resnet.fit(np.expand_dims(train, axis = -1), to_categorical(train_labels),
                        validation_data = [np.expand_dims(val, axis = -1), to_categorical(val_labels)],
                        epochs = 25, batch_size = 5, optimizer = opt2, save = True) #train model
    
def main():
    ''''''

    train_models() # train resnet and bayesian resnet

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