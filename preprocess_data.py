import matplotlib.pyplot as plt
import numpy as np

from glob import glob #for parsing paths
from skimage import io #for loading images
from skimage.transform import resize #for downsizing
from skimage.color import rgb2gray #for converting to grayscale
from imblearn.over_sampling import RandomOverSampler, SMOTE #for class balance

from utils import plotClassDist, plotSamples, saveData, loadData

def load_images(downsize = True, downsize_shape = (256, 256)):
    '''Load the images, downsize if true, convert to greyscale, combine and generate labels.'''
    
    #get list of image paths
    files_normal = glob('data\\chest_xray\\normal\\*.jpeg')
    files_pneumonia = glob('data\\chest_xray\\pneumonia\\*.jpeg')
    
    #load images, downsize, and append to numpy arrays
    normal = np.zeros((len(files_normal), *downsize_shape), dtype = np.uint8)
    for i, fname in enumerate(files_normal):
        print('({}/{}) Loading image {} ...'.format(i, len(files_normal), fname))
        #downsize image if true
        if downsize:
            img = resize(io.imread(fname), downsize_shape)
        #some images are RGB for some reason so convert to grayscale if that is the case
        if img.ndim == 3:
            img = rgb2gray(img)
        normal[i] = np.asarray(img * 255.0).astype(np.uint8)
    
    pneumonia = np.zeros((len(files_pneumonia), *downsize_shape), dtype = np.uint8)
    for i, fname in enumerate(files_pneumonia):
        print('({}/{}) Loading image {} ...'.format(i, len(files_pneumonia), fname))
        #downsize image if true
        if downsize:
            img = resize(io.imread(fname), downsize_shape)
        #some images are RGB for some reason so convert to grayscale if that is the case
        if img.ndim == 3:
            img = rgb2gray(img)
        pneumonia[i] = np.asarray(img * 255.0).astype(np.uint8)
    
    #combine into one array and generate labels {0:normal, 1:pneumonia}
    print('Combing data and generating labels ...', end = ' ')
    n, p = normal.shape[0], pneumonia.shape[0]
    data = np.zeros((n + p, *downsize_shape), dtype = np.uint8)
    labels = np.zeros((n + p, ), dtype = np.uint8)
    data[0:n, :, :] = normal
    data[n:n + p, :, :] = pneumonia
    labels[0:n] = 0
    labels[n:n + p] = 1
    
    print('Done!')
    return data, labels

def randomOversample(data, labels):
    '''Randomly oversample the data. We need to flatten the images first and the deflatten when done.'''
    shp = data.shape
    data = data.reshape(shp[0], shp[1] * shp[2]) #flatten data
    ros = RandomOverSampler(random_state=42)
    data, labels = ros.fit_resample(data, labels) #oversample
    data = data.reshape(data.shape[0], shp[1], shp[2]) #de-flatten data
    return data, labels
    
def smoteOversample(data, labels):
    ''' Oversample the data using SMOTE. We need to flatten the images first and the deflatten when done.'''
    shp = data.shape
    data = data.reshape(shp[0], shp[1] * shp[2]) #flatten data
    smt = SMOTE(random_state=42)
    data, labels = smt.fit_resample(data, labels) #oversample using smote
    data = data.reshape(data.shape[0], shp[1], shp[2]) #de-flatten data
    return data, labels

def main():
    #load the JPEG images or the npz file
    #data, labels = load_images() #un-comment if want to load JPEG images into numpy
    data, labels = loadData('data/data.npz')  #load images from npz
    
    #check imbalance
    print('Data Shape:', data.shape, 'Labels Shape:', labels.shape)
    plotClassDist(labels)
    
    #random oversample and recheck balance
    ovs_data, ovs_labels = randomOversample(data, labels)
    print('OVS Data Shape:', ovs_data.shape, 'OVS Labels Shape:', ovs_labels.shape)
    plotClassDist(ovs_labels)
    
    #oversample using smote and recheck balance
    smt_data, smt_labels = smoteOversample(data, labels)
    print('SMT Data Shape:', smt_data.shape, 'SMT Labels Shape:', smt_labels.shape)
    plotClassDist(smt_labels)
    
    #save untouched and oversampled data as npz
    saveData('data/data.npz', data, labels)
    saveData('data/ros_data.npz', ovs_data, ovs_labels)
    saveData('data/smt_data.npz', smt_data, smt_labels)
    
    
if __name__ == '__main__':
    main()