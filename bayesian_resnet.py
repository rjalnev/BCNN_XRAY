import numpy as np
import os

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from utils import loadData, saveData, plotSamples, plotClassDist

class BayesianResNet():
    
    def __init__(self, input_shape, num_classes, num_filters = 32, kernel = 3, blocks = [1, 2, 2, 1],
                 load = False, directory = './model/'):
        ''''''
        # Note: error in using load_model for conv2dflipout and denseflipout layers even when passing them as custom objectes.
        # Issues are described in detail in these posts: https://github.com/tensorflow/probability/issues/289
        #                                                https://github.com/tensorflow/probability/issues/325
        #                                                https://github.com/tensorflow/probability/issues/516
        # Easy fix is to save_weights_only = True for ModelCheckpoint and load weights instead of model.
        self.kl_weight = tf.Variable(1.0) #scaling factor for the kl divergence function, initialized to 1.0
        self.model = self.build_model(input_shape, num_classes, num_filters, kernel, blocks);
        if load: #load the weights from h5 file and set current epoch
            self.model.load_weights(directory + 'bayesian.h5', by_name = True)
            self.current_epoch = np.genfromtxt(directory + 'bayesian_hist.csv', delimiter = ',', skip_header = 1).shape[0]
        else: #new model, current epoch is zero
            self.current_epoch = 0
    
    def build_model(self, input_shape, num_classes, num_filters, kernel, blocks):
        ''''''
        #kl divergence scaled by number of samples
        kl_divergence_function = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) * self.kl_weight)
        
        def residual_block(x, num_filters, kernel, stride, index):
            out = tfp.layers.Convolution2DFlipout(num_filters, kernel, stride, padding = 'same', name = 'C{}'.format(index),
                                                  kernel_divergence_fn = kl_divergence_function)(x)
            index += 1
            out = ReLU(name = 'RELU{}'.format(index-1))(out)
            out = BatchNormalization(name = 'BN{}'.format(index-1))(out)
            out = tfp.layers.Convolution2DFlipout(num_filters, kernel, 1, padding = 'same', name = 'C{}'.format(index),
                                                  kernel_divergence_fn = kl_divergence_function)(out)
            index += 1
            if stride == 2: #Need to downsample input by have to match dimensions
                x = tfp.layers.Convolution2DFlipout(num_filters, 1, stride, padding = 'same', name = 'C_DS{}'.format(index-3),
                                                    kernel_divergence_fn = kl_divergence_function)(x)
            out = Add(name = 'ADD{}'.format(index-3))([x, out])
            out = ReLU(name = 'RELU{}'.format(index-1))(out)
            out = BatchNormalization(name = 'BN{}'.format(index-1))(out)
            return out, index

        x = Input(shape = input_shape, name = 'input')
        
        out = tfp.layers.Convolution2DFlipout(num_filters, kernel, 1, padding = 'same', name = 'C1',
                                              kernel_divergence_fn = kl_divergence_function)(x)
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
        out = tfp.layers.DenseFlipout(num_classes, activation='softmax', name = 'output',
                                      kernel_divergence_fn = kl_divergence_function)(out)
        
        model = Model(x, out, name = 'bayesian_resnet')
        model.summary()
        return model
    
    def fit(self, X, Y, validation_data = None, epochs = 10, batch_size = 32, optimizer = 'adam',
            save = False, directory = './model/'):
        ''''''
        if save: # set callback functions if saving model
            if not os.path.exists(directory): os.makedirs(directory)
            mpath = directory + "bayesian.h5"
            hpath = directory + 'bayesian_hist.csv'
            if validation_data is None:
                checkpoint = ModelCheckpoint(filepath = mpath, monitor = 'loss', verbose = 0, save_best_only = True, save_weights_only = True)
            else:
                checkpoint = ModelCheckpoint(filepath = mpath, monitor = 'val_loss', verbose = 0, save_best_only = True, save_weights_only = True)
            cvs_logger = CSVLogger(hpath, separator = ',', append = True)
            callbacks = [cvs_logger, checkpoint]
        else:
            callbacks = None
        
        self.kl_weight.assign(tf.cast(1.0 / X.shape[0], dtype = tf.float32)) #scale kl_divergence function by number of training samples
        self.model.compile(optimizer, 'categorical_crossentropy', ['accuracy'],
                           experimental_run_tf_function = False) #compile model
        self.model.fit( X, Y, validation_data = validation_data,
                        epochs = epochs, batch_size = batch_size, shuffle = True,
                        initial_epoch = self.current_epoch,
                        callbacks = callbacks,
                        verbose = 1 ) #fit model
                        
        self.current_epoch += epochs
        
    def predict(self, x, mc_steps = 50):
        ''''''
        def entropy(p):
            return -1 * np.sum(np.log(p + 1e-15) * p, axis=0)
        
        pred = np.asarray([self.model.predict(x) for _ in range(mc_steps)])
        mean_pred = np.mean(pred, axis = 0)
        entropy = np.apply_along_axis(entropy, axis = 1, arr = mean_pred) 
        return np.argmax(mean_pred, axis = 1), entropy   