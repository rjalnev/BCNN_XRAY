import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from utils import loadData, saveData, plotSamples, plotClassDist

class ResNet():
    
    def __init__(self, load = False, directory = './model/', input_shape = None, num_classes = None,
                 num_filters = 32, kernel = 3, blocks = [1, 2, 2, 1]):
        ''''''
        if load:
            self.model = load_model(directory + 'resnet.h5')
            self.current_epoch = np.genfromtxt(directory + 'resnet_hist.csv', delimiter = ',', skip_header = 1).shape[0]
        else:
            assert (input_shape is not None or num_classes is not None), 'The arguments input_shape and num_filters must be specified if not loading a model.'
            self.model = self.build_model(input_shape, num_classes, num_filters, kernel, blocks);
            self.current_epoch = 0
    
    def build_model(self, input_shape, num_classes, num_filters, kernel, blocks):
        ''''''
        def residual_block(x, num_filters, kernel, stride, index):
            out = Conv2D(num_filters, kernel, stride, padding = 'same', name = 'C{}'.format(index))(x)
            index += 1
            out = ReLU(name = 'RELU{}'.format(index-1))(out)
            out = BatchNormalization(name = 'BN{}'.format(index-1))(out)
            out = Conv2D(num_filters, kernel, 1, padding = 'same', name = 'C{}'.format(index))(out)
            index += 1
            if stride == 2: #Need to downsample input by half to match dimensions
                x = Conv2D(num_filters, 1, stride, padding = 'same', name = 'C_DS{}'.format(index-3))(x)
            out = Add(name = 'ADD{}'.format(index-3))([x, out])
            out = ReLU(name = 'RELU{}'.format(index-1))(out)
            out = BatchNormalization(name = 'BN{}'.format(index-1))(out)
            return out, index

        x = Input(shape = input_shape, name = 'input')
        
        out = Conv2D(num_filters, kernel, 1, padding = 'same', name = 'C1')(x)
        out = ReLU(name = 'RELU1')(out)
        out = BatchNormalization(name = 'BN1')(out)
        
        index = 2
        for i in range(len(blocks)):
            for j in range(blocks[i]):
                if (i!=0 and j==0): stride = 2
                else: stride = 1
                out, index = residual_block(out, num_filters, kernel, stride, index)
            num_filters *= 2
                
        out = AveragePooling2D(name = 'avgPool')(out)
        out = Flatten(name = 'flatten')(out)
        out = Dense(num_classes, activation='softmax', name = 'output')(out)
        
        model = Model(x, out, name = 'resnet')
        model.summary()
        return model
    
    def fit(self, X, Y, validation_data = None, epochs = 10, batch_size = 32, optimizer = 'adam',
            save = False, directory = './model/'):
        ''''''
        if save: # set callback functions if saving model
            if not os.path.exists(directory): os.makedirs(directory)
            mpath = directory + "resnet.h5"
            hpath = directory + 'resnet_hist.csv'
            if validation_data is None:
                checkpoint = ModelCheckpoint(filepath = mpath, monitor = 'loss', verbose = 0, save_best_only = True)
            else:
                checkpoint = ModelCheckpoint(filepath = mpath, monitor = 'val_loss', verbose = 0, save_best_only = True)
            cvs_logger = CSVLogger(hpath, separator = ',', append = True)
            callbacks = [cvs_logger, checkpoint]
        else:
            callbacks = None
        
        self.model.compile(optimizer, 'categorical_crossentropy', ['accuracy']) #compile model
        self.model.fit( X, Y, validation_data = validation_data,
                        epochs = epochs, batch_size = batch_size, shuffle = True,
                        initial_epoch = self.current_epoch,
                        callbacks = callbacks,
                        verbose = 1 ) #fit model
                        
        self.current_epoch += epochs
        
    def predict(self, x):
        ''''''
        return np.argmax(self.model.predict(x), axis = 1)