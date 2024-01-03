import os
import numpy as np
from river import metrics as river_metrics
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import zipfile
import shutil
import scipy.io as sio
from scipy.io import loadmat

class datawork:
    @staticmethod
    def sort_by_class(X_train, y_train, num_classes):
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

    @staticmethod
    def assure_path_exists(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    @staticmethod
    def shuffle(X_train, y_train, seed=66):
        np.random.seed(seed)
        num_pts = X_train.shape[0]
        ind = np.arange(num_pts)
        np.random.shuffle(ind)
        X_train = X_train[ind]
        y_train = y_train[ind]
        return X_train, y_train

    @staticmethod
    def compute_accuracies(y_pred, y_true):
        acc = river_metrics.accuracy_score(y_true, y_pred) * 100
        return acc

    @staticmethod
    def normalize_data(X):
        norm = np.kron(np.ones((X.shape[1], 1)), np.linalg.norm(X, axis=1)).T
        X = X / norm
        return X

    @staticmethod
    def load_cub200(data_folder, experiment):
        train_data_path = os.path.join(data_folder, 'cub200_resnet50_train.mat')
        test_data_path = os.path.join(data_folder, 'cub200_resnet50_test.mat')

        train = sio.loadmat(train_data_path)
        test = sio.loadmat(test_data_path)

        X_train = train['X']
        y_train = train['y'].ravel()
        X_test = test['X']
        y_test = test['y'].ravel()

        X_train = datawork.normalize_data(X_train)
        X_test = datawork.normalize_data(X_test)

        if experiment == 'class_iid':
            X_train, y_train = datawork.sort_by_class(X_train, y_train, 200)
        elif experiment == 'iid':
            X_train, y_train = datawork.shuffle(X_train, y_train)
        else:
            raise NotImplementedError('Experiment type not supported.')
        return X_train, y_train, X_test, y_test

    @staticmethod
    def get_dataset_params(dataset):
        if dataset == 'cub200':
            capacity_list = [2, 4, 8, 16]
        else:
            raise NotImplementedError('Dataset not supported.')
        return capacity_list

    @staticmethod
    def unzip_file(zip_file_path, extracted_dir):
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
            print(f'Successfully unzipped {zip_file_path} to {extracted_dir}')
        except Exception as e:
            print(f'Error unzipping {zip_file_path}: {str(e)}')

    @staticmethod
    def move_files(src_dir, dst_dir):
        try:
            shutil.move(src_dir, dst_dir)
            print(f'Successfully moved files from {src_dir} to {dst_dir}')
        except Exception as e:
            print(f'Error moving files: {str(e)}')

    #@staticmethod
    #def load_cub200(path, experiment):
        #train = sio.loadmat(path + '/' + 'cub200_resnet50_train.mat')
        #test = sio.loadmat(path + '/' + 'cub200_resnet50_test.mat')

        #X_train = train['X']
        #y_train = train['y'].ravel()
        #X_test = test['X']
        #y_test = test['y'].ravel()

        #X_train = DataUtils.normalize_data(X_train)
        #X_test = DataUtils.normalize_data(X_test)

       