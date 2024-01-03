import os
import numpy as np
import sklearn.metrics
import scipy
from river import metrics as river_metrics
from scipy.io import loadmat
import torch
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets

import os
import zipfile
from scipy.io import loadmat
import numpy as np


def sort_by_class(X_train, y_train, num_classes):
    """
    Organize data by class.
    :param X_train: training data
    :param y_train: training labels
    :param num_classes: number of classes
    :return: organized training data and labels
    """
    X = np.zeros_like(X_train)
    y = np.zeros_like(y_train)
    c = 0
    for i in range(num_classes):
        idx = y_train == i
        Xi = X_train[idx]
        yi = y_train[idx]
        num = Xi.shape[0]
        X[c:c + num] = Xi
        y[c:c + num] = yi
        c += num
    return X, y

def assure_path_exists(dir):
    """
    Make sure path exists for saving.
    :param dir: the path
    :return:
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

def shuffle(X_train, y_train, seed=66):
    """
    Shuffle data.
    :param X_train: training data
    :param y_train: training labels
    :return: shuffled data
    """
    np.random.seed(seed)
    num_pts = X_train.shape[0]
    ind = np.arange(num_pts)
    np.random.shuffle(ind)
    X_train = X_train[ind]
    y_train = y_train[ind]
    return X_train, y_train

def compute_accuracies(y_pred, y_true):
    acc = river_metrics.accuracy_score(y_true, y_pred) * 100
    return acc

def normalize_data(X):
    """
    Make each feature have unit norm (divide by L2 norm).
    :param X: data
    :return: normalized data
    """
    norm = np.kron(np.ones((X.shape[1], 1)), np.linalg.norm(X, axis=1)).T
    X = X / norm
    return X



import os
import glob
from PIL import Image
import numpy as np

def load_cub200(path, experiment):
    """
    Set up the CUB-200 dataset and experiment.
    :param path: path to cub features
    :param experiment: string of experiment type
    :return: training and testing data and labels
    """
    train = loadmat(path + '/' + 'cub200_resnet50_train.mat')
    test = loadmat(path + '/' + 'cub200_resnet50_test.mat')

    X_train = train['X']
    y_train = train['y'].ravel()
    X_test = test['X']
    y_test = test['y'].ravel()

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    if experiment == 'class_iid':
        X_train, y_train = sort_by_class(X_train, y_train, 200)
    elif experiment == 'iid':
        X_train, y_train = shuffle(X_train, y_train)
    else:
        raise NotImplementedError('Experiment type not supported.')
    return X_train, y_train, X_test, y_test

    
def get_dataset_params(dataset):
    """
    Get the capacities to test for the particular dataset.
    :param dataset: string of dataset
    :return: list of capacities
    """
    if dataset == 'cub200':  # Updated for MNIST
        capacity_list = [2, 4, 8, 16]  # Example capacities for MNIST
    else:
        raise NotImplementedError('Dataset not supported.')
    return capacity_list









