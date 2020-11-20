# adaboost imports

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import numpy as np
from feature_extraction_image_preprocessing import getFeature

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

def run_simulations(features, params):
    model = AdaBoostClassifier(
        n_estimators=params['n_estimators'],
        base_estimator=DecisionTreeClassifier(max_depth=1),
        random_state=42              
    )
    header = ["feature_name", "cv_0_score","cv_1_score", "cv_2_score", "cv_3_score", "cv_4_score"]
    for feature in features:
        X,y = split_data_to_xy(getFeature(feature))        
        print(feature, adaboost_score(model, X, y))


if __name__ == "__main__":
    print("Hello AdaBoost")
    params = {}
    params['n_estimators'] = 20
    params['base_estimator'] = DecisionTreeClassifier(max_depth=1)
    params['algorithm'] = "SAMME.R"    
    features = ["gray_scale", "mean_pixel", "extracting_edge_vertical", "extracting_edge_horizontal", "hog" ]
    run_simulations(features, params)