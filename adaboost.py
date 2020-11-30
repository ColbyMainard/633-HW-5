# adaboost imports

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import numpy as np
from feature_extraction_image_preprocessing import getFeature

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK


def adaboost_score(clf, X, y):
    cv_results = cross_validate(clf, X, y)
    return cv_results['test_score']

def split_data_to_xy(train_data):
    y,X = train_data.T 
    ll = []
    for i in X:
        ll.append(i)
    X = np.array(ll)
    return X, y

def run_simulations(params):
    model = AdaBoostClassifier(
        n_estimators=params['n_estimators'],
        base_estimator=params['base_estimator'],
        random_state=42              
    )
    # print(combined_X)
    # print(combined_X.shape)
    X = params['X']
    y = params['y']

    result = np.mean(adaboost_score(model, X, y))
    return {
            'model': model, 
            'loss': 1-result,
            'status':STATUS_OK
            }


def get_data(features):
    combined_X = np.zeros(250)
    for feature in features:
        X,y = split_data_to_xy(getFeature(feature))        
        combined_X = np.column_stack((combined_X,X))
    combined_X  = combined_X[:, 1:]
    return combined_X, y



if __name__ == "__main__":
    dt = DecisionTreeClassifier(max_depth=1)
    nb = GaussianNB()
    search_space = {
        'n_estimators': hp.choice('n_estimators', [10,20, 40, 80, 160, 320]),
        'base_estimator': hp.choice('base_estimator', [dt,  nb]),        
    }
    features = ["gray_scale", 
            "mean_pixel", 
            "extracting_edge_vertical", 
            "extracting_edge_horizontal", 
            "hog"
            ]
    X, y = get_data(features)
    search_space['X'] = X
    search_space['y'] = y

    hypopt_trials = Trials()
    best_params  = fmin(run_simulations, 
            search_space, 
            algo=tpe.suggest, 
            max_evals=10,
            trials=hypopt_trials)

    with open("output.txt", "w") as f:
        print(best_params, file=f)
        print(hypopt_trials.best_trial['result'], file=f)
