import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages, un-comment if want to see but they annoy me

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.utils import to_categorical #use for one-hot encoding labels
from tensorflow.keras.optimizers import Adam #optimizer

from resnet import ResNet
from utils import loadData

def main():
    ''''''
    #load data from npz
    data, labels = loadData('data/ros_data.npz')

    #resnet with 4 stages with [1, 2, 2, 1] number of residual blocks
    resnet = ResNet(input_shape = (256, 256, 1), num_classes = 2) #init model
    opt = Adam(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False) #setup optimizer
    resnet.fit(np.expand_dims(data, axis = -1), to_categorical(labels), batch_size = 5, optimizer = opt, save=False) #train model
    
    #bayesian resnet
    
    
if __name__ == '__main__':
    main()