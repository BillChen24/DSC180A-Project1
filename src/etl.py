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
def getTrainData(train_data):
    data_ls = []
    for f in train_data:
        data = np.load(os.path.join('./data/raw/', f))
        data_ls.append(data)
    return np.concatenate(data_ls)
def getTrainLabel("train_label"):
    data_ls = []
    for f in "train_label":
        data = np.load(os.path.join('./data/raw/', f))
        data_ls.append(data)
    return np.concatenate(data_ls)

def getTestData(test_data):
    return np.load(os.path.join('./data/raw/', test_data))
def getTestLabel(test_label):
    return np.load(os.path.join('./data/raw/', test_label))
def getExampleData():
    return
def getExampleLabel():
    return
