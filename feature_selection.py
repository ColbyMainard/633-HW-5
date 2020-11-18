import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from feature_extraction_image_preprocessing import *

def wrapperFeatureSelect(x_train, y_train):
    model = LogisticRegression()

    rfe_engine = RFE(model, 5)
    rfe_result = rfe_engine.fit(x_train, y_train)
    
    print("Num Features: %s"        % (rfe_result.n_features_))
    print("Selected Features: %s"   % (rfe_result.support_))
    print("Feature Ranking: %s"     % (rfe_result.ranking_))
    

def filterFeatureSelect(x_train, y_train):
    test = SelectKBest(score_func=mutual_info_classif, k=5)
    filter_result = test.fit(x_train, y_train)

    # Summarize scores
    np.set_printoptions(precision=3)
    print(filter_result.scores_)

    features = filter_result.transform(x_train)

    # Summarize selected features
    print(features[0:5,:])


def reduceFeatures(x_train):
    pca = PCA(n_components=250)
    principalComponents = pca.fit_transform(x_train)
    return principalComponents


if __name__ == "__main__":
    y_data          = np.load('train_data.npy')
    x_gray_scale    = getFeature("gray_scale")
    x_mean_pixel    = getFeature("mean_pixel")
    x_edge_vertical = getFeature("extracting_edge_vertical")
    x_edge_horizon  = getFeature("extracting_edge_horizontal")
    x_hog           = getFeature("hog")

    x_train = []
    y_train = []

    print("Combining image features...",end="")
    for index in range(len(x_gray_scale)):
        x_train.append(np.concatenate((x_gray_scale[index][1],    \
                                       x_mean_pixel[index][1],    \
                                       x_edge_horizon[index][1],  \
                                       x_edge_vertical[index][1], \
                                       x_hog[index][1])))
        for y_index in range(len(y_data)):
            if x_gray_scale[index][0] == y_data[y_index][0]:
                y_train.append(int(y_data[y_index][4]))
    print("done")

    print("Reducing image features with PCA...",end="")
    principalComponents = reduceFeatures(x_train)
    print("done")
    x_train = np.array(principalComponents)
    print(np.shape(x_train))

    # print("Selecting features with RFE (wrapper)...",end="")
    # wrapperFeatureSelect(x_train, y_train)
    # print("done")

    print("Selecting features with LinearDiscriminantAnalysis (filter)...",end="")
    filterFeatureSelect(x_train, y_train)
    print("done")