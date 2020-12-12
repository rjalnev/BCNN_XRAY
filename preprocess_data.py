import matplotlib.pyplot as plt
import numpy as np

from glob import glob #for parsing paths
from skimage import io #for loading images
from skimage.transform import resize #for downsizing
from skimage.color import rgb2gray #for converting to grayscale
from sklearn.model_selection import train_test_split #for splitting dataset
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
        print('({}/{}) Loading image {} ...'.format(i+1, len(files_normal), fname))
        #downsize image if true
        if downsize:
            img = resize(io.imread(fname), downsize_shape)
        #some images are RGB for some reason so convert to grayscale if that is the case
        if img.ndim == 3:
            img = rgb2gray(img)
        normal[i] = np.asarray(img * 255.0).astype(np.uint8)
    
    pneumonia = np.zeros((len(files_pneumonia), *downsize_shape), dtype = np.uint8)
    for i, fname in enumerate(files_pneumonia):
        print('({}/{}) Loading image {} ...'.format(i+1, len(files_pneumonia), fname))
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
    #saveData('data/data.npz', data, labels) #un-comment if want to save loaded JPEG images as npz for faster loading later
    data, labels = loadData('data/data.npz')  #load images from npz, much faster then loading from JPEGS each time
    
    #split dataset into train, val, and test ... split into 70/30 and then 70/15/15
    print('Splitting data into 70/15/15 train, val, and test sets.')
    train, testval, train_labels, testval_labels = train_test_split(data, labels, test_size = 0.30, random_state = 42, shuffle = True, stratify = labels)
    test, val, test_labels, val_labels = train_test_split(testval, testval_labels, test_size = 0.50, random_state = 42, shuffle = True, stratify = testval_labels)
    del data, labels, testval, testval_labels #free up memory
    
    #check imbalance
    print('Train Shape:', train.shape, 'Train Labels Shape:', train_labels.shape)
    print('Validation Shape:', val.shape, 'Validation Labels Shape:', val_labels.shape)
    print('Test Shape:', test.shape, 'Test Labels Shape:', test_labels.shape)
    plotClassDist(train_labels, 'Train Class Distribution')
    plotClassDist(val_labels, 'Validation Class Distribution')
    plotClassDist(test_labels, 'Test Class Distribution')
    
    #random oversample train set and recheck balance
    ovs_data, ovs_labels = randomOversample(train, train_labels)
    print('OVS Data Shape:', ovs_data.shape, 'OVS Labels Shape:', ovs_labels.shape)
    plotClassDist(ovs_labels, 'Train Class Distribution (ROS)')
    
    #oversample using smote on train set and recheck balance
    smt_data, smt_labels = smoteOversample(train, train_labels)
    print('SMT Data Shape:', smt_data.shape, 'SMT Labels Shape:', smt_labels.shape)
    plotClassDist(smt_labels, 'Train Class Distribution (SMOTE)')
    
    #save untouched and oversampled data as npz along with val and test
    saveData('data/train.npz', train, train_labels)
    saveData('data/ros_data.npz', ovs_data, ovs_labels)
    saveData('data/smt_data.npz', smt_data, smt_labels)
    saveData('data/val.npz', val, val_labels)
    saveData('data/test.npz', test, test_labels)
    
    
if __name__ == '__main__':
    main()