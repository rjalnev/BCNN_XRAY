import matplotlib.pyplot as plt
import numpy as np

def plotClassDist(labels, title=None):
    '''Plot the class distribution.'''
    classes, count = np.unique(labels, return_counts=True)
    plt.bar(classes, count, tick_label = classes.astype(np.int))
    if title == None:
        plt.title('Class Distribution')
    else:
        plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

def plotSamples(data, labels, columns = 5, rows = 5):
    '''Plot a grid of randomly sampled images of size columns by rows.'''
    class_names = ['Normal', 'Pneumonia']
    plt.figure(figsize=(10,10))
    for i in range(rows * columns):
        idx = np.random.randint(0, data.shape[0])
        plt.subplot(columns, rows, i + 1)
        plt.xticks([]); plt.yticks([]);
        plt.grid(False)
        plt.imshow(data[idx, :, :], cmap=plt.cm.gray)
        plt.xlabel(class_names[int(labels[idx])])
    plt.show()
    
def saveData(path, data, labels):
    '''Save the data and labels as npz.'''
    print('Saving data to {} ...'.format(path), end = ' ')
    keywords = {'data': data, 'labels': labels}
    np.savez(path, **keywords)
    print('Done!')

def loadData(path):
    '''Load the npz file and return data and labels.'''
    print('Loading data from {} ...'.format(path), end = ' ')
    data = np.load(path)
    print('Done!')
    return data['data'], data['labels']
    
def calculate_accuracy(pred_labels, true_labels):
    '''Given the predicted labels and true labels calculate accuracy.'''
    return np.sum(pred_labels == true_labels) / pred_labels.shape[0]