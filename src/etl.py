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

    return np.concatenate(train_data_ls), np.concatenate(train_label_ls), np.concatenate(test_data_ls), np.concatenate(test_label_ls)

def getExampleData():
    return
def getExampleLabel():
    return
