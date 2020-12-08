import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten

from utils import loadData, saveData, plotSamples, plotClassDist

class BayesianResNet():
    
    def __init__(self, load = False, directory = './model/', input_shape = None, num_classes = None,
                 kl_weight = None, num_filters = 32, kernel = 3, blocks = [1, 2, 2, 1]):
        ''''''
        if load:
            self.model = load_model(directory + 'model.h5')
            self.current_epoch = np.genfromtxt(directory + 'model_history.csv', delimiter = ',', skip_header = 1).shape[0]
        else:
            assert (input_shape is not None or num_classes is not None or kl_weight is not None), 'The arguments input_shape, num_classes, and kl_weight must be specified if not loading a model.'
            assert (kl_weight < 1 and kl_weight > 0), 'The argument kl_weight should be 1/num_samples to correctly scale the kl divergence.'
            self.model = self.build_model(input_shape, num_classes, kl_weight, num_filters, kernel, blocks);
            self.current_epoch = 0
    
    def build_model(self, input_shape, num_classes, kl_weight, num_filters, kernel, blocks):
        ''''''
        #kl divergence scaled by number of samples
        kl_divergence_function = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) * tf.cast(kl_weight, dtype = tf.float32))
        
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
        
        model = Model(x, out, name = 'resnet')
        model.summary()
        return model
    
    def fit(self, X, Y, validation_data = None, epochs = 10, batch_size = 32, optimizer = 'adam',
            save = False, directory = './model/'):
        ''''''
        if save: # set callback functions if saving model
            if not os.path.exists(directory): os.makedirs(directory)
            mpath = directory + "wifinet.h5"
            hpath = directory + 'wifinet_history.csv'
            if validation_data is None:
                checkpoint = ModelCheckpoint(filepath = mpath, monitor = 'loss', verbose = 0, save_best_only = True)
            else:
                checkpoint = ModelCheckpoint(filepath = mpath, monitor = 'val_loss', verbose = 0, save_best_only = True)
            cvs_logger = CSVLogger(hpath, separator = ',', append = True)
            callbacks = [cvs_logger, checkpoint]
        else:
            callbacks = None
        
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