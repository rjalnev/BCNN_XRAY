import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages, un-comment if want to see but they annoy me

import matplotlib.pyplot as plt
import numpy as np
from time import time

from tensorflow.keras.utils import to_categorical #use for one-hot encoding labels
from tensorflow.keras.optimizers import SGD #optimizer

from resnet import ResNet
from bayesian_resnet import BayesianResNet
from utils import loadData, calculate_metrics, gen_fake_data, print_metrics

def get_models(load_saved_model = False):
    '''Create or load models.'''
    if load_saved_model:
        resnet = ResNet(load = True) #load resnet
        bayesian_resnet = BayesianResNet(input_shape = (256, 256, 1), num_classes = 2, load = True) #load bayesian resnet
    else:
        resnet = ResNet(input_shape = (256, 256, 1), num_classes = 2) #init resnet
        bayesian_resnet = BayesianResNet(input_shape = (256, 256, 1), num_classes = 2) #init bayesian resnet
    return resnet, bayesian_resnet


def train_models(resnet, bayesian_resnet, num_epochs = 10):
    '''Train models until num_epochs reached.'''
    #load training and validation datasets 
    train, train_labels = loadData('data/ros_data.npz')
    val, val_labels = loadData('data/val.npz')
    
    opt = SGD(learning_rate = 1e-3) #setup optimizer
    
    resnet.fit(np.expand_dims(train, axis = -1), to_categorical(train_labels),
              validation_data = [np.expand_dims(val, axis = -1), to_categorical(val_labels)],
               epochs = num_epochs, batch_size = 5, optimizer = opt, save = True) #train model
    
    bayesian_resnet.fit(np.expand_dims(train, axis = -1), to_categorical(train_labels),
                        validation_data = [np.expand_dims(val, axis = -1), to_categorical(val_labels)],
                        epochs = num_epochs, batch_size = 5, optimizer = opt, save = True) #train model


def test_model(model, data, labels, mc_steps = None):
    '''Test models and return probalities, predicted labels, entropy, and metrics.'''
    start_time = time()
    if mc_steps is None:
        pred, pred_labels, entropy = model.predict(np.expand_dims(data, axis = -1))
    else:
        pred, pred_labels, entropy = model.predict(np.expand_dims(data, axis = -1), mc_steps = mc_steps)
    elapsed_time = time() - start_time;
    metrics = calculate_metrics(pred_labels, labels) + [elapsed_time]
    
    print('{} ~ Accuracy: {:.4f} ~ Time: {:.2f} seconds'.format(model.model.name, metrics[0], metrics[4]))
    return pred, pred_labels, entropy, metrics


def main():
    resnet, bayesian_resnet = get_models(load_saved_model = True) #load or create the models
    #train_models(resnet, bayesian_resnet, num_epochs = 10) #train resnet and bayesian resnet until number of epochs trained reaches num_epochs
    
    #test models and print metrics
    test, test_labels = loadData('data/test.npz');
    rpred, rpred_labels, rentropy, rmetrics = test_model(resnet, test, test_labels)
    bpred, bpred_labels, bentropy, bmetrics = test_model(bayesian_resnet, test, test_labels)
    
    print_metrics('resnet', rmetrics)
    print_metrics('bayesian', bmetrics)

    #eliminate predictions with high entropy and print new metrics
    idx1 = np.argwhere(rentropy < 0.50)
    idx2 = np.argwhere(bentropy < 0.50)
    print_metrics('resnet', calculate_metrics(rpred_labels[idx1], test_labels[idx1]))
    print_metrics('bayesian', calculate_metrics(bpred_labels[idx2], test_labels[idx2]))

    #generate fake data and create a mixed datasets
    fake, fake_labels = gen_fake_data(300)
    mixed = np.concatenate((test, fake), axis = 0)
    mixed_labels = np.concatenate((test_labels, fake_labels), axis = 0)
    
    #run prediction on mixed dataset and get metrics
    rpred, rpred_labels, rentropy, rmetrics = test_model(resnet, mixed, mixed_labels)
    bpred, bpred_labels, bentropy, bmetrics = test_model(bayesian_resnet, mixed, mixed_labels)
    
    print_metrics('resnet', rmetrics)
    print_metrics('bayesian', bmetrics)
    
    #eliminate predictions with high entropy on mixed dataset and print new metrics
    idx1 = np.argwhere(rentropy < 0.50)
    idx2 = np.argwhere(bentropy < 0.50)
    print_metrics('resnet', calculate_metrics(rpred_labels[idx1], mixed_labels[idx1]))
    print_metrics('bayesian', calculate_metrics(bpred_labels[idx2], mixed_labels[idx2]))


if __name__ == '__main__':
    main()