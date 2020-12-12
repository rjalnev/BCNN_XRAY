import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import precision_recall_fscore_support

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
    
def calculate_metrics(pred_labels, true_labels):
    '''Given the predicted labels and true labels calculate accuracy, precision, recall, and f-score.'''
    accuracy = np.sum(pred_labels == true_labels) / pred_labels.shape[0]
    precision, recall, fscore, _ = precision_recall_fscore_support(true_labels, pred_labels, average = 'binary')
    return [accuracy, precision, recall, fscore]
    
def gen_fake_data(num_images):
    '''Generate fake data.'''
    images = np.random.randint(low = 0, high = 256,  size = (num_images, 256, 256))
    labels = np.random.randint(low = 0, high = 2, size = num_images)
    return images, labels
    
def print_metrics(model_name, metrics):
    '''Format and print the metrics.'''
    if len(metrics) == 5:
        str_metrics = '{:8} | Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | FScore {:.4f} | Time: {:.2f} seconds'
    else:
        str_metrics = '{:8} | Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | FScore {:.4f}'
    print(str_metrics.format(model_name, *metrics))