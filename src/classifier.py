import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import pickle

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
                print('finish training experiment:', exp_id)
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
              modeltype):

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
                        'criterion': 'gini', 'max_depth': 50, 'max_features': 'sqrt', 
                        'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 
                        'min_samples_leaf': 10, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 
                        'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 
                        'random_state': 28, 'verbose': 0, 'warm_start': False}
            clf=clf.set_params(**best_hyper)
        
    clf.fit(X_train, y_train)
    return clf

def clf_predict(clf, X_test, y_test, predictions_fp):
    predictions = clf.predict(X_test)
    out = pd.concat([pd.Series(y_test), pd.Series(predictions)], axis=1)
    out.to_csv(predictions_fp, header = ['label', 'prediction'])
    print(classification_report(y_test, predictions))
    return predictions



def accuracy_report(y_test, y_pred):
    return classification_report(y_pred, y_test)
