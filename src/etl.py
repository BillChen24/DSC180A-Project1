# import pickle
# def get_data(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict[b'data']
#
# def get_label(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict[b'labels']

import numpy as np
import os
# def getTrainData(train_data):
#     data_ls = []
#     for f in train_data:
#         data = np.load(os.path.join('./data/raw/', f))
#         data_ls.append(data)
#     return np.concatenate(data_ls)
#
# def getTrainLabel(train_label):
#     data_ls = []
#     for f in train_label:
#         data = np.load(os.path.join('./data/raw/', f))
#         data_ls.append(data)
#     return np.concatenate(data_ls)
#
# def getTestData(test_data):
#     return np.load(os.path.join('./data/raw/', test_data))
# def getTestLabel(test_label):
#     return np.load(os.path.join('./data/raw/', test_label))
def getData(outdir, train_data, train_label, test_data, test_label):
    train_data_ls = []
    for f in train_data:
        data = np.load(os.path.join(outdir, f))
        train_data_ls.append(data)

    train_label_ls = []
    for f in train_label:
        data = np.load(os.path.join(outdir, f))
        train_label_ls.append(data)

    test_data_ls = []
    for f in test_data:
        data = np.load(os.path.join(outdir, f))
        test_data_ls.append(data)

    test_label_ls = []
    for f in test_label:
        data = np.load(os.path.join(outdir, f))
        test_label_ls.append(data)
        
    animal_labels = [2, 4, 5, 6, 7]
    
    X_train = np.concatenate(train_data_ls)
    y_train = np.concatenate(train_label_ls)
    X_test = np.concatenate(test_data_ls)
    y_test = np.concatenate(test_label_ls)
    
    train_query = [l in animal_labels for l in y_train]
    X_train = X_train[train_query]
    y_train = y_train[train_query]
    test_query = [l in animal_labels for l in y_test]
    X_test = X_test[test_query]
    y_test = y_test[test_query]

    return X_train, y_train, X_test, y_test

def getExampleData():
    return
def getExampleLabel():
    return
