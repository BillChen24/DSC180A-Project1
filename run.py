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
    print(targets)
    # -------------- DATA LOADING -----------------
    data_to_use=targets[0]
    test_trial=False
    if data_to_use =='test':
        test_trial=True
        with open('config/test-params.json', 'r') as fh:
            data_params = json.load(fh)
        X_train, y_train, X_test, y_test = getData(**data_params)
        print('load test data')
        print('train_data:', X_train.shape[0], 'test data:', X_test.shape[0])
        # with open('config/model-params.json', 'r') as fh:
        #     model_params = json.load(fh)
        # clf = clf_build(X_train, y_train, "RandomForestClassifier")
        # clf_predict(clf, X_test, y_test, "data/out/intial_preds_test.csv")
        # print('initial model on test')
        # X_train_gray = grayscale(X_train, "X_train_gray")
        # X_test_gray = grayscale(X_test, "X_test_gray")
        # print('do gray scale')
        

    elif data_to_use == 'all':
        #TODO: load all data
        with open('config/data-params.json', 'r') as fh:
            data_params = json.load(fh)
        X_train, y_train, X_test, y_test = getData(**data_params)
        print('load all data')
        print(X_train.shape, X_test.shape)
        print('train_data:', X_train.shape[0], 'test data:', X_test.shape[0])
        
        
    # Train / Read Model:
    model = train_model( X_train, y_train, X_test, y_test ,targets = targets)    
    
    #filter, transform
    data_paths=filte_transport_data(X_test, y_test, test = test_trial)
    
    #make predction and get prediction tables
    
    if 'prediction_table' in targets:
        
        prediction_paths=create_pred_table(data_paths, model, y_test)
    else:
        reports = create_pred_acc_report(data_paths, model, y_test)
        
        
    if 'plot' in targets:
        ...
        
        
        

#     #train classifier on training data, predict on testing data and write prediction
#     if 'model' in targets:
#         with open('config/model-params.json', 'r') as fh:
#             model_params = json.load(fh)
#         clf = clf_build(X_train, y_train, "RandomForestClassifier")
#         clf_predict(clf, X_test, y_test, "data/out/intial_preds.csv")

#     #color transform on all data, write data in data/temp
#     if 'color_transform' in targets:
#         X_train_gray = grayscale(X_train, "X_train_gray")
#         X_test_gray = grayscale(X_test, "X_test_gray")

#     #use classifier on color transformed data
#     if 'test1' in targets:
#         clf_predict(clf, X_test_gray, y_test, "data/out/grayscale_preds.csv")

#     #train OT on given data
#     if 'ot_build' in targets:
#         Xs, Xt = sample_color(X_train_gray,X_train, 5000)
#         cot = color_ot_build(Xs, Xt, ot.da.EMDTransport())
# 	#apply OT on given data
#     if 'ot_transform' in targets:
#         X_test_ot = color_ot_transform(X_test_gray, cot)

# 	#test, validate result, produce output
#     if 'test2' in targets:
#         clf_predict(clf, X_test_ot, y_test, "data/out/ot_preds.csv")

def train_model(train_data, train_label,test_data, test_label,targets = None):
    '''
    detect if word tune in targest, if so, tune model
    if not, read in best model saved / train default one
    '''
    if 'tune' in targets:
        model = classifier.model_selelection_RandomForestClassifier(train_data, train_label, 
                                                 depths=[10, 20, 30], 
                                                 estimators=[100],
                                                 min_samples_leaf = [5,10,20,40, 50])
        
    else:
    # Read Model
        model=clf_build(train_data, train_label)
    # Produce report by class
    print('Performance on Training Set')
    clf_predict(model,train_data, train_label, 'result/prediction1.csv')
    print('Performance on Test Set')
    clf_predict(model,test_data, test_label, 'result/prediction1.csv')
    # prediction on filtered data, save the result to result
    prediction_df(train_data, train_label, 
                         model = model,
                        outdf_path = 'result/prediction_train.csv')
    prediction_df(test_data, test_label, 
                         model = model,
                        outdf_path = 'result/prediction_original.csv')
    print('Prediction Table on Test set saved at', 'result/prediction_original.csv')
    return model


def filte_transport_data(X, y, test=False):
    #return path to the folder containing the filter, trans_without, trains_with class
    #DO NOT CHANGE KEY NAME!
    if test==False:
        return_dic={'filtered':'data/temp/filtered/filtered.npy',
                   'trans_without_class': ['data/temp/transformer_without_class/transformed.npy',
                                          'data/temp/transformer_without_class/transformed_1000.npy'],
                   'trans_with_class': [f'data/temp/transformer_with_class/transformed_{i}.npy' for i in [2,4,5,6,7]],
                   'trans_within_class': [f'data/temp/transformer_within_class/transformed_{i}.npy' for i in [2,4,5,6,7]]}
        if os.path.exists(return_dic['trans_without_class'][0]):
            return return_dic
        #else: #if no data exist, has to filter, train transporter, and transport data
    else: # if apply to test data set, read from test data set
        print('Use test trans data')
        return_dic={'filtered':'test/temp/filtered/filtered.npy',
                   'trans_without_class':[ 'test/temp/transformer_without_class/transformed.npy'],
                   'trans_with_class': [f'test/temp/transformer_with_class/transformed_{i}.npy' for i in [2,4,5,6,7]],
                   'trans_within_class': [f'test/temp/transformer_within_class/transformed_{i}.npy' for i in [2,4,5,6,7]]}
        if os.path.exists(return_dic['trans_without_class'][0]):
            return return_dic                                                  

def create_pred_table(data_fp_dict, model, test_label, result_fp = 'result/'):
    '''
    data_fp_dict: dictionary contain keys: filtered, trans_without_class, trans_with_class, trans_within_class
    takes in pretrained model
    '''
    return_dict=data_fp_dict.copy()
    #Filter WithOUT transformer
#     if data_fp_dict['filtered'] is not None:
#         fp=data_fp_dict['filtered']
#         test_data=np.load(fp)
#         fp=fp.split('/')
#         outdf_path = 'result/'+fp[-2]+'_'+fp[-1].replace('npy', 'csv')
#         prediction_df(test_data, test_label, 
#                          model = model,
#                         outdf_path = outdf_path)
#         return_dict['filtered'] = outdf_path
#         print('prediction table save to', outdf_path)
    
    
    #WITHOUT class Transformed
    if data_fp_dict['trans_without_class'] is not None:
        for fp_idx in range(len(data_fp_dict['trans_without_class'])):
            fp = data_fp_dict['trans_without_class'][fp_idx]
            test_data=np.load(fp)
            fp=fp.split('/')
            outdf_path = 'result/'+fp[-2]+'_'+fp[-1].replace('npy', 'csv')
            prediction_df(test_data, test_label, 
                             model = model,
                            outdf_path = outdf_path)
            return_dict['trans_without_class'][fp_idx] = outdf_path
            print('prediction table save to', outdf_path)
        
    #WITH CLASS TRANSFORMED
    if data_fp_dict['trans_with_class'] is not None:
        for fp_idx in range(len(data_fp_dict['trans_with_class'])):
            fp = data_fp_dict['trans_with_class'][fp_idx]
            test_data=np.load(fp)
            fp=fp.split('/')
            outdf_path = 'result/'+fp[-2]+'_'+fp[-1].replace('npy', 'csv')
            prediction_df(test_data, test_label, 
                             model = model,
                            outdf_path = outdf_path)
            return_dict['trans_with_class'][fp_idx] = outdf_path
            print('prediction table save to', outdf_path)
    #WITH CLASS TRANSFORMED
    # if data_fp_dict['trans_within_class'] is not None:
    #     for fp_idx in range(len(data_fp_dict['trans_within_class'])):
    #         fp = data_fp_dict['trans_within_class'][fp_idx]
    #         test_data=np.load(fp)
    #         fp=fp.split('/')
    #         outdf_path = 'result/'+fp[-2]+'_'+fp[-1].replace('npy', 'csv')
    #         prediction_df(test_data, test_label, 
    #                          model = model,
    #                         outdf_path = outdf_path)
    #         return_dict['trans_within_class'][fp_idx] = outdf_path
    #         print('prediction table save to', outdf_path)
        
    return return_dict


def create_pred_acc_report(data_fp_dict, model, test_label):
    '''
    data_fp_dict: dictionary contain keys: filtered, trans_without_class, trans_with_class, trans_within_class
    takes in pretrained model
    '''
    return_dict = data_fp_dict.copy()
    # #Filter WithOUT transformer
    #  if data_fp_dict['filtered'] is not None:
    #     fp=data_fp_dict['filtered']
    #     test_data=np.load(fp)
    #     fp=fp.split('/')
    #     outdf_path = 'result/'+fp[-2]+'_'+fp[-1].replace('npy', '')
    #     print('Performance on Data ', outdf_path)
    #     pred = model.predict(test_data)
    #     report = accuracy_report(test_label, pred)
    #     return_dict['filtered'] = report
    
    
    #WITHOUT class Transformed
    if data_fp_dict['trans_without_class'] is not None:
        for fp_idx in range(len(data_fp_dict['trans_without_class'])):
            fp = data_fp_dict['trans_without_class'][fp_idx]
            test_data=np.load(fp)
            
            fp=fp.split('/')
            outdf_path = 'result/'+fp[-2]+'_'+fp[-1].replace('npy', '')
            print('Performance on Data ', outdf_path)
            pred = model.predict(test_data)
            report = accuracy_report(test_label, pred)
            return_dict['trans_without_class'][fp_idx] = report
        
    #WITH CLASS TRANSFORMED
    if data_fp_dict['trans_with_class'] is not None:
        for fp_idx in range(len(data_fp_dict['trans_with_class'])):
            fp = data_fp_dict['trans_with_class'][fp_idx]
            test_data=np.load(fp)
            fp=fp.split('/')
            outdf_path = 'result/'+fp[-2]+'_'+fp[-1].replace('npy', '')
            print('Performance on Data ', outdf_path)
            pred = model.predict(test_data)
            report = accuracy_report(test_label, pred)
            return_dict['trans_with_class'][fp_idx] = report
           
    #WITH CLASS TRANSFORMED
#     if data_fp_dict['trans_within_class'] is not None:
#         for fp_idx in range(len(data_fp_dict['trans_within_class'])):
#             fp = data_fp_dict['trans_within_class'][fp_idx]
#             test_data=np.load(fp)
#             fp=fp.split('/')
#             outdf_path = 'result/'+fp[-2]+'_'+fp[-1].replace('npy', '')
#             print('Performance on Data ', outdf_path)
#             pred = model.predict(test_data)
#             report = accuracy_report(test_label, pred)
#             return_dict['trans_within_class'][fp_idx] = report
            
        
    return return_dict
    
    
    
# # RUN Way1
#import argparse
#parser = argparse.ArgumentParser(description = 'DSC180A Project')
#parser.add_argument("target", type=str)
#main(parser.parse_args())

## Run Way2
if __name__ == '__main__': #if run from command line
    targets = sys.argv[1:]
    main(targets)
    