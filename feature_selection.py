import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from feature_extraction_image_preprocessing import *


def wrapperFeatureSelect(x_train, y_train, num_features):
    model = LogisticRegression()

    rfe_engine = RFE(model, num_features)
    rfe_result = rfe_engine.fit(x_train, y_train)

    print("Num Features: %s" % rfe_result.n_features_)
    print("Selected Features: %s" % rfe_result.support_)
    print("Feature Ranking: %s" % rfe_result.ranking_)

    # Shrink training data to only those features selected
    features = rfe_result.transform(x_train)
    print("shape of features = ", end="")
    print(np.shape(features))
    return features


def filterFeatureSelect(x_train, y_train, num_features):
    test = SelectKBest(score_func=mutual_info_classif, k=num_features)
    filter_result = test.fit(x_train, y_train)

    # # Summarize scores
    # np.set_printoptions(precision=3)
    # print(filter_result.scores_)

    # Shrink training data to only those features selected
    features = filter_result.transform(x_train)
    print("shape of features = ", end="")
    print(np.shape(features))
    return features


def reduceFeatures(x_train):
    pca = PCA(n_components=250)
    principalComponents = pca.fit_transform(x_train)
    return principalComponents


def testModel(x_data, y_data):
    # model = LogisticRegression(penalty='none')
    # x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.20,random_state=0)
    # model.fit(x_train, y_train)
    # y_pred=model.predict(x_test)
    # cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    # print(cnf_matrix)
    # print("accuracy = " + str(metrics.accuracy_score(y_test, y_pred)))

    accuracies = []
    x_parts, y_parts = splitData(x_data, y_data)  

    for index in range(5):
        x_test = x_parts[index]
        y_test = y_parts[index]

        x_train = np.array([])
        y_train = np.array([])
        for part in range(5):
            if part != index:
                if x_train.size == 0:
                    x_train = x_parts[part]
                    y_train = y_parts[part]
                else:
                    x_train = np.concatenate((x_train, x_parts[part]))
                    y_train = np.concatenate((y_train, y_parts[part]))

        print(np.shape(x_train))
        model = LogisticRegression(penalty='none')
        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        accuracies.append(metrics.accuracy_score(y_test, y_pred))
    print("Average Accuracy = ",end="")
    print(sum(accuracies)/len(accuracies))


def splitData(x_data, y_data):
    x_data1, x_part1, y_data1, y_part1 = train_test_split(x_data ,y_data, test_size=0.20,random_state=0)
    x_data2, x_part2, y_data2, y_part2 = train_test_split(x_data1,y_data1,test_size=0.25,random_state=0)
    x_data3, x_part3, y_data3, y_part3 = train_test_split(x_data2,y_data2,test_size=0.33,random_state=0)
    x_part4, x_part5, y_part4, y_part5 = train_test_split(x_data3,y_data3,test_size=0.50,random_state=0)

    x_parts = [x_part1, x_part2, x_part3, x_part4, x_part5]
    y_parts = [y_part1, y_part2, y_part3, y_part4, y_part5]

    return x_parts, y_parts


if __name__ == "__main__":
    x_gray_scale = getFeature("gray_scale")
    x_mean_pixel = getFeature("mean_pixel")
    x_edge_vertical = getFeature("extracting_edge_vertical")
    x_edge_horizon = getFeature("extracting_edge_horizontal")
    x_hog = getFeature("hog")

    x_train = []
    y_train = []
    print("\n========================================================================")
    print("Combining image features...")
    for index in range(len(x_gray_scale)):
        x_train.append(np.concatenate((x_gray_scale[index][1],
                                       x_mean_pixel[index][1],
                                       x_edge_horizon[index][1],
                                       x_edge_vertical[index][1],
                                       x_hog[index][1])))
        y_train.append(x_gray_scale[index][0])

    print("\n========================================================================")
    print("Reducing image features with PCA...")
    principalComponents = reduceFeatures(x_train)
    x_train = np.array(principalComponents)
    print(np.shape(x_train))

    print("\n========================================================================")
    print("Selecting features with RFE (wrapper)...")
    x_train_wrapper = wrapperFeatureSelect(x_train, y_train, 250)
    print("\nTest feature set selected by RFE on training data...")
    testModel(x_train_wrapper, y_train)

    print("\n========================================================================")
    print("Selecting features with Mutual Info Classification (filter)...")
    x_train_filter = filterFeatureSelect(x_train, y_train, 250)
    print("\nTest feature set selected by filter method on training data...")
    testModel(x_train_filter, y_train)
    
    