import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

def models():

    mdls = {
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier()
    }

    return mdls

def clf_build(X_train,
              y_train,
              modeltype):

    clf = models()[modeltype] # get model from dict
    #clf = clf(**params) # instantiate model w/given params
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
