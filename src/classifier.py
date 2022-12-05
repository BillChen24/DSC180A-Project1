import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

def models():

    mdls = {
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier(random_state=28)
    }

    return mdls


def model_selelection_RandomForestClassifier(train_data_X, train_data_Y, 
                                             depths=[5,10,20,50], 
                                             estimators=[50,100,200],
                                             min_samples_leaf = [1, 5, 20, 100]
                                            ):
    '''
    Takes in train_X, Y data  and lists of hyperparameter to experiment
    Return the Model with best hyperparameter setting from selection
    Also save the model as best_model.sav in result
    '''
    #initialize the result table
    col_names = ['exp_id','max_depth','max_features','min_samples_leaf','n_estimators','result']
    results = pd.DataFrame(columns =col_names )

    # train, val split
    train_X, val_X, train_Y, val_Y = train_test_split(
        train_data_X,train_data_Y, test_size=0.7, train_size=0.3, random_state=28
        )


    exp_id=0
    
    for max_depth in depths:
        for est in estimators:
            for minleaf in min_samples_leaf:
                rfc=models()["RandomForestClassifier"]

                hyperparam={'max_depth':max_depth,
                       'max_features':'sqrt','min_samples_leaf':minleaf,
                       'n_estimators':est}
                #print(hyperparam)
                rfc=rfc.set_params(**hyperparam)
                rfc.fit(train_X,train_Y)

                hyperparam['exp_id'] = exp_id
                hyperparam['result'] = rfc.score(val_X,val_Y)

                results.loc[exp_id]=hyperparam
                print('finish training experiment:', exp_id, hyperparam['result'])
                exp_id+=1
                
    best_result = (results
                  .sort_values(by='result').iloc[-1]
                  
                 )
    print(best_result)
    best_hyper = best_result[['max_depth','max_features','min_samples_leaf','n_estimators']].to_dict()
    rfc=models()["RandomForestClassifier"]
    rfc=rfc.set_params(**best_hyper)
    rfc = rfc.fit(train_data_X,train_data_Y)
    # To Save Model
    filename = 'result/best_rfc_model.sav'
    pickle.dump(rfc, open(filename, 'wb'))
    
    #result = loaded_model.score(X_test, Y_test)
    print(rfc.get_params())
        #identify best hyperparameters
    
    #return model with best hyper
    return rfc


def clf_build(X_train,y_train,
              modeltype='RandomForestClassifier'):

    clf = models()[modeltype] # get model from dict
    #clf = clf(**params) # instantiate model w/given params
    
    if modeltype == 'RandomForestClassifier':
        # for Rand Forest Classifier: Check if pretrained best model exist.
        #if exist use best model
        try:
            filename = 'result/best_rfc_model.sav'
            clf = pickle.load(open(filename, 'rb'))
            print('Read Pre Trained Best Model')
            return clf
        except:
            print('Non pre trained model, Train by pre determined best hyperparameters')
            best_hyper={'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 
                        'criterion': 'gini', 'max_depth': 20, 'max_features': 'sqrt', 
                        'min_samples_leaf': 5, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 
                        'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 
                        'random_state': 28, 'verbose': 0, 'warm_start': False}
            clf=clf.set_params(**best_hyper)
        
    clf.fit(X_train, y_train)
    return clf

def clf_predict(clf, X_test, y_test, predictions_fp):
    predictions = clf.predict(X_test)
    out = pd.concat([pd.Series(y_test), pd.Series(predictions)], axis=1)
    out.to_csv(predictions_fp, header = ['label', 'prediction'])
    accuracy_report(y_test, predictions)
    return predictions



def accuracy_report(y_test, y_pred):
    report =  classification_report(y_pred, y_test)
    print(report)
    return report

def report_votes(X_in, model = 'result/best_rfc_model.sav'):
    '''
    Takes in Randomforest Model or path to read the model
    X: array of size Nx3072
    
    output: array of Nx5, with N equals to the row of given X
    '''
    #Read in model if given path to best
    if type(model) == str:
        try:
            model = pickle.load(open(model, 'rb'))
            print('Read Pre Trained Best Model')
            
        except:
            print('Fail to read pretrained model')
    else:
        print('use input model')
        
    # make prediction
    n_size = X_in.shape[0]
    
    #count votes in the tree
    result_arr=np.zeros((n_size, 5))
    for tree in model.estimators_:
        result = tree.predict(X_in)
        for idx in range(n_size):
            result_arr[idx][int(result[idx])]+=1
            
    return result_arr


def prediction_df(X, y,model='result/best_rfc_model.sav', outdf_path = 'result/prediction_table.csv'):
    '''
    Take in data X, data label
    model: model or path to the best model
    
    return dataframe with iamge true label, predictions and prediction vote count
    '''
    n_size = X.shape[0]
    #initial table
    df = pd.DataFrame(columns = [ 'true_label','predicted_label','predict_vote'])
    df['predict_vote'] = np.arange(n_size)
    
    # vote count:
    vote_count=report_votes(X, model = model)
    
    #get predicted label
    predictions=np.array([2,4,5,6,7])[vote_count.argmax(axis=1)]
    
    df['true_label'] =y
    df['predicted_label'] = predictions
    def put_in_count(idx):
        return vote_count[idx].tolist()
    df['predict_vote'] = df['predict_vote'].apply(put_in_count)
    df['max_count'] = vote_count.max(axis=1).astype(int)
    df.to_csv(outdf_path)
    return df