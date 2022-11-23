#!/usr/bin/env python

import sys
import os
import json
from src.etl import *
from src.classifier import *
from src.color_transform import *
from src.optimal_transport import *

#
# parser.add_argument('--batch_size', type = int, default = 1,
#         help = 'input batch size for training (default: 1)')

# parser.add_argument('--z_score',dest = 'normalization', action='store_const',default = data.min_max_normalize, const = data.z_score_normalize,help = 'use z-score normalization on the dataset, default is min-max normalization')


def main(targets):
	#target=targets.target
    data_to_use=targets[0]

    if data_to_use =='test':
        with open('config/test-params.json', 'r') as fh:
            data_params = json.load(fh)
        X_train, y_train, X_test, y_test = getData(**data_params)
        print('load test data')
        with open('config/model-params.json', 'r') as fh:
            model_params = json.load(fh)
        clf = clf_build(X_train, y_train, "RandomForestClassifier")
        clf_predict(clf, X_test, y_test, "data/out/intial_preds_test.csv")
        print('initial model on test')
        X_train_gray = grayscale(X_train, "X_train_gray")
        X_test_gray = grayscale(X_test, "X_test_gray")
        print('do gray scale')
        

    elif data_to_use == 'all':
        #TODO: load all data
        with open('config/data-params.json', 'r') as fh:
            data_params = json.load(fh)
        X_train, y_train, X_test, y_test = getData(**data_params)
        print('load all data')

    #elif data_to_use == 'clean':
	#TODO: clean all the generate result files

    #    print('clear to clean repo')
    #else:
    #    print('No clear instruction')

    #train classifier on training data, predict on testing data and write prediction
    if 'model' in targets:
        with open('config/model-params.json', 'r') as fh:
            model_params = json.load(fh)
        clf = clf_build(X_train, y_train, "RandomForestClassifier")
        clf_predict(clf, X_test, y_test, "data/out/intial_preds.csv")

    #color transform on all data, write data in data/temp
    if 'color_transform' in targets:
        X_train_gray = grayscale(X_train, "X_train_gray")
        X_test_gray = grayscale(X_test, "X_test_gray")

    #use classifier on color transformed data
    if 'test1' in targets:
        clf_predict(clf, X_test_gray, y_test, "data/out/grayscale_preds.csv")

    #train OT on given data
    if 'ot_build' in targets:
        Xs, Xt = sample_color(X_train_gray,X_train, 5000)
        cot = color_ot_build(Xs, Xt, ot.da.EMDTransport())
	#apply OT on given data
    if 'ot_transform' in targets:
        X_test_ot = color_ot_transform(X_test_gray, cot)

	#test, validate result, produce output
    if 'test2' in targets:
        clf_predict(clf, X_test_ot, y_test, "data/out/ot_preds.csv")


# # RUN Way1
#import argparse
#parser = argparse.ArgumentParser(description = 'DSC180A Project')
#parser.add_argument("target", type=str)
#main(parser.parse_args())

## Run Way2
if __name__ == '__main__': #if run from command line
    targets = sys.argv[1:]
    main(targets)
