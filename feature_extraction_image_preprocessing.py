# image-preprocessor.py
# This file will contain all necessary methods and data to implement a.-b.iii.
# This translates to image preprocessing, visual feature extraction, feature exploration, and feature selection
# All the above functionality should be made available in methods/classes as to allow for easy access in other files

import statistics
import os
import cv2
import numpy as np
import matplotlib.pyplot as plot
from natsort import natsorted
from skimage.filters import prewitt_h, prewitt_v
from skimage.feature import hog
from skimage import exposure
from scipy.stats import entropy


def getImageInfo(mode):
    total_test_images = 0

    if mode == 'train':
        path = TRAIN_IMAGE_DIR
    else:
        path = TEST_IMAGE_DIR

    # IMPORTANT
    # we need to natsort the list otherwise we will not get the correct index
    # because python traverse the list as 1, 11, 12, 13, ...
    for image in natsorted(os.listdir(path)):
        image_name = image
        full_path_to_image = os.path.join(path, image)
        image = cv2.imread(full_path_to_image)

        # square crop and resize then save the new image
        resize(image, image_name, mode=mode)

        total_test_images += 1


def resize(image, image_name, mode='train'):
    resize_resolution = (600, 600)

    height, width, channel = image.shape
    center_height = height // 2
    center_width = height // 2

    if center_height > resize_resolution[0] or center_width > resize_resolution[1]:
        cropped_image = image[center_height - resize_resolution[0] // 2:center_height + resize_resolution[0] // 2,
                        center_width - resize_resolution[1] // 2:center_width + resize_resolution[1] // 2]
    else:
        cropped_image = image[:resize_resolution[0], :resize_resolution[1]]

    # resize image
    resized = cv2.resize(cropped_image, resize_resolution, interpolation=cv2.INTER_AREA)

    # cv2.imshow("Normal image", image)
    # cv2.imshow("Resized image", resized)
    # cv2.waitKey(0)

    if mode == 'train':
        image_name = os.path.join(RESIZE_TRAIN_IMAGE_DIR, image_name)
    else:
        image_name = os.path.join(RESIZE_TEST_IMAGE_DIR, image_name)
    cv2.imwrite(image_name, resized)


class ImagePreprocessor:
    def __init__(self):
        self.resolution = (600, 600)
        self.gray_scale = []
        self.mean_pixel = []
        self.extracting_edge = []
        self.hog = []

    def featureExtraction(self):
        # for train data
        for image in os.listdir(RESIZE_TRAIN_IMAGE_DIR):
            image_name = image
            full_path_to_image = os.path.join(RESIZE_TRAIN_IMAGE_DIR, image)
            gray_image = cv2.imread(full_path_to_image, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(full_path_to_image)
            self.grayScaleFeature(gray_image, image_name)
            self.meanPixelValueOfChannels(image, image_name)
            self.extractingEdgeFeature(gray_image, image_name)
            self.hogFeature(image, image_name)

        # convert to numpy array to save
        self.gray_scale = np.array(self.gray_scale)
        self.mean_pixel = np.array(self.mean_pixel)
        self.extracting_edge = np.array(self.extracting_edge)
        self.hog = np.array(self.hog)

        np.save('gray_scale_feature.npy', self.gray_scale)
        np.save('mean_pixel_feature.npy', self.mean_pixel)
        np.save('extracting_edge_feature.npy', self.extracting_edge)
        np.save('hog_feature.npy', self.hog)

        # for test data
        self.gray_scale = []
        self.mean_pixel = []
        self.extracting_edge = []
        self.hog = []

        for image in os.listdir(RESIZE_TEST_IMAGE_DIR):
            image_name = image
            full_path_to_image = os.path.join(RESIZE_TEST_IMAGE_DIR, image)
            gray_image = cv2.imread(full_path_to_image, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(full_path_to_image)
            self.grayScaleFeature(gray_image, image_name)
            self.meanPixelValueOfChannels(image, image_name)
            self.extractingEdgeFeature(gray_image, image_name)
            self.hogFeature(image, image_name)

        # convert to numpy array to save
        self.gray_scale = np.array(self.gray_scale)
        self.mean_pixel = np.array(self.mean_pixel)
        self.extracting_edge = np.array(self.extracting_edge)
        self.hog = np.array(self.hog)

        np.save('test_gray_scale_feature.npy', self.gray_scale)
        np.save('test_mean_pixel_feature.npy', self.mean_pixel)
        np.save('test_extracting_edge_feature.npy', self.extracting_edge)
        np.save('test_hog_feature.npy', self.hog)

    def grayScaleFeature(self, gray_image, image_name):
        gray_scale_features = np.reshape(gray_image, self.resolution[0] * self.resolution[1])
        self.gray_scale.append((image_name, gray_scale_features))

    def meanPixelValueOfChannels(self, image, image_name):
        feature_matrix = np.zeros(self.resolution)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                feature_matrix[i][j] = (int(image[i, j, 0]) + int(image[i, j, 1]) + int(image[i, j, 2])) / 3
        features = np.reshape(feature_matrix, self.resolution[0] * self.resolution[1])
        self.mean_pixel.append((image_name, features))

    def extractingEdgeFeature(self, image, image_name):
        # calculating horizontal edges using prewitt kernel
        edges_prewitt_horizontal = prewitt_h(image)

        # calculating vertical edges using prewitt kernel
        edges_prewitt_vertical = prewitt_v(image)

        # horizontal for image 1 is self.extracting_edge[0][0]
        # vertical for image 1 is self.extracting_edge[0][1]
        edge_features = (edges_prewitt_horizontal, edges_prewitt_vertical)
        self.extracting_edge.append((image_name, edge_features))

    def hogFeature(self, image, image_name):
        # creating hog features
        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, multichannel=True)

        # rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        self.hog.append((image_name, hog_image_rescaled))


def getFeature(feature):
    """
        4 possible features, * means more important

        gray_scale or test_gray_scale
        mean_pixel * or test_mean_pixel
        extracting_edge * or test_extracting_edge
        hog * or test_hog


        I think we should use a combination of:
            mean_pixel with extracting_edge
            mean_pixel with hog

        The load data will be in the form of: (image_name, feature array)
        So to get the label of the feature, you need to get the name from train_data.npy
        The name is the first element
        The label is the last element
    """
    if feature == 'gray_scale':
        return np.load('gray_scale_feature.npy', allow_pickle=True)

    if feature == 'test_gray_scale':
        return np.load('test_gray_scale_feature.npy', allow_pickle=True)

    if feature == 'mean_pixel':
        return np.load('mean_pixel_feature.npy', allow_pickle=True)

    if feature == 'test_mean_pixel':
        return np.load('test_mean_pixel_feature.npy', allow_pickle=True)

    if feature == 'extracting_edge':
        return np.load('extracting_edge_feature.npy', allow_pickle=True)

    if feature == 'test_extracting_edge':
        return np.load('test_extracting_edge_feature.npy', allow_pickle=True)

    if feature == 'hog':
        return np.load('hog_feature.npy', allow_pickle=True)

    if feature == 'test_hog':
        return np.load('test_hog_feature.npy', allow_pickle=True)


class DataPoint:
    def __init__(self):
        self.data = []
        self.total_data = 250
        self.age = []
        self.gender = []
        self.location = []

    def getData(self):
        data_file = open('train.csv', 'r')
        for index, data in enumerate(data_file):
            # skip first line
            if index:
                # split by , will cause a problem where the location also have , in them
                data = data.split(',')

                # the train data we have sometimes doesnt have all the value so we need to take care
                # of that by removing every other feature based on the index in the data
                # and what left is the location

                image_name = data[0]
                data.pop(0)

                gender = data[0]
                data.pop(0)
                self.gender.append(gender)

                age = data[0]
                data.pop(0)
                try:
                    self.age.append(int(age))
                except ValueError:
                    self.age.append(0)

                # remove the \n at the end of the line
                data[-1] = data[-1].split('\n')[0]
                label = data[-1]
                data.pop()

                # we only need to care about the city
                location = data[-1]
                self.location.append(location)
                self.data.append([image_name, gender, age, location, label])
        self.fixMissingValue()

    def stringListMedian(self, array):
        half_length = len(array) // 2
        return array[half_length]

    def fixMissingValue(self):
        for data in self.data:
            gender = data[1]
            age = data[2]
            location = data[3]
            if gender == '':
                data[1] = self.stringListMedian(self.gender)

            if age == '':
                data[2] = int(statistics.median(self.age))

            if location == '':
                data[3] = self.stringListMedian(self.location)
        self.data = np.array(self.data)
        np.save('train_data.npy', self.data)

    def ageHistogram(self):
        age_dict_infected = {}
        age_dict_uninfected = {}

        undefined_infected = 0
        undefined_uninfected = 0

        for i in self.data:
            label = int(i[-1])
            try:
                age = int(i[2])
                if label:
                    if age in age_dict_infected:
                        age_dict_infected[age] += 1
                    else:
                        age_dict_infected[age] = 1
                else:
                    if age in age_dict_uninfected:
                        age_dict_uninfected[age] += 1
                    else:
                        age_dict_uninfected[age] = 1
            except ValueError:
                # empty age value
                if label:
                    undefined_infected += 1
                else:
                    undefined_uninfected += 1

        infected_age = list(age_dict_infected.keys())
        infected_count = list(age_dict_infected.values())
        uninfected_age = list(age_dict_uninfected.keys())
        uninfected_count = list(age_dict_uninfected.values())

        infected = sum(infected_count) / self.total_data
        uninfected = sum(uninfected_count) / self.total_data
        age_entropy = entropy([infected, uninfected])

        # print(age_entropy)
        plot.title(f'Age Entropy: {age_entropy}')
        plot.bar(infected_age, list(infected_count), label="Infected", color='red')
        plot.bar(uninfected_age, list(uninfected_count), label="Uninfected", color='lime')
        plot.legend(loc='upper right')
        plot.show()

        order = [1, 2]
        data = [undefined_infected, undefined_uninfected]
        labels = ['Infected', 'Uninfected']
        plot.bar(order, data)
        plot.xticks(order, labels)
        plot.show()

    def genderHistogram(self):
        male_dict_infected = {}
        male_dict_uninfected = {}
        female_dict_infected = {}
        female_dict_uninfected = {}

        undefined_infected = 0
        undefined_uninfected = 0

        for i in self.data:
            label = int(i[-1])
            gender = str(i[1])
            if gender == 'M':
                if label:
                    if gender in male_dict_infected:
                        male_dict_infected[gender] += 1
                    else:
                        male_dict_infected[gender] = 1
                else:
                    if gender in male_dict_uninfected:
                        male_dict_uninfected[gender] += 1
                    else:
                        male_dict_uninfected[gender] = 1
            elif gender == 'F':
                if label:
                    if gender in female_dict_infected:
                        female_dict_infected[gender] += 1
                    else:
                        female_dict_infected[gender] = 1
                else:
                    if gender in female_dict_uninfected:
                        female_dict_uninfected[gender] += 1
                    else:
                        female_dict_uninfected[gender] = 1
            else:
                # empty gender value
                if label:
                    undefined_infected += 1
                else:
                    undefined_uninfected += 1

        male_infected_key = list(male_dict_infected.keys())
        male_infected_count = list(male_dict_infected.values())
        male_uninfected_key = list(male_dict_uninfected.keys())
        male_uninfected_count = list(male_dict_uninfected.values())

        female_infected_key = list(female_dict_infected.keys())
        female_infected_count = list(female_dict_infected.values())
        female_uninfected_key = list(female_dict_uninfected.keys())
        female_uninfected_count = list(female_dict_uninfected.values())

        male = (sum(male_infected_count) + sum(male_uninfected_count)) / self.total_data
        female = (sum(female_infected_count) + sum(female_uninfected_count)) / self.total_data

        gender_entropy = entropy([male, female])

        plot.title(f'Gender Entropy: {gender_entropy}')
        plot.bar(male_infected_key, list(male_infected_count), label="Male Infected", color='red')
        plot.bar(male_uninfected_key, list(male_uninfected_count), label="Male Uninfected", color='lime')
        plot.bar(female_infected_key, list(female_infected_count), label="Female Infected", color='black')
        plot.bar(female_uninfected_key, list(female_uninfected_count), label="Female Uninfected", color='cyan')
        plot.legend(loc='upper right')
        plot.show()

        order = [1, 2]
        data = [undefined_infected, undefined_uninfected]
        labels = ['Infected', 'Uninfected']
        plot.bar(order, data)
        plot.xticks(order, labels)
        plot.show()

    def locationHistogram(self):
        location_dict_infected = {}
        location_dict_uninfected = {}

        undefined_infected = 0
        undefined_uninfected = 0

        for i in self.data:
            label = int(i[-1])
            location = str(i[3])
            if location != '':
                # remove the " at the end
                if location[-1] == '\"':
                    location = location[:-1]

                if label:
                    if location in location_dict_infected:
                        location_dict_infected[location] += 1
                    else:
                        location_dict_infected[location] = 1
                else:
                    if location in location_dict_uninfected:
                        location_dict_uninfected[location] += 1
                    else:
                        location_dict_uninfected[location] = 1
            else:
                # empty location value
                if label:
                    undefined_infected += 1
                else:
                    undefined_uninfected += 1

        infected_location = list(location_dict_infected.keys())
        infected_count = list(location_dict_infected.values())
        uninfected_location = list(location_dict_uninfected.keys())
        uninfected_count = list(location_dict_uninfected.values())

        infected = sum(infected_count) / self.total_data
        uninfected = sum(uninfected_count) / self.total_data
        location_entropy = entropy([infected, uninfected])

        # print(location_entropy)
        plot.title(f'Location Entropy: {location_entropy}')
        plot.bar(infected_location, list(infected_count), label="Infected", color='red')
        plot.bar(uninfected_location, list(uninfected_count), label="Uninfected", color='lime')
        location, labels = plot.xticks()
        plot.setp(labels, rotation=70, horizontalalignment='right')
        plot.legend(loc='upper right')
        plot.show()

        order = [1, 2]
        data = [undefined_infected, undefined_uninfected]
        labels = ['Infected', 'Uninfected']
        plot.bar(order, data)
        plot.xticks(order, labels)
        plot.show()

    def visualizeFeature(self):
        # load the save data
        self.data = np.load('train_data.npy')

        self.ageHistogram()

        self.genderHistogram()

        self.locationHistogram()


def updateLabelToFeature():
    dataset = np.load('train_data.npy')
    gray_scale_feature = getFeature('gray_scale')
    mean_pixel_feature = getFeature('mean_pixel')
    extracting_edge_feature = getFeature('extracting_edge')
    hog_feature = getFeature('hog')

    for i in range(250):
        name = dataset[i][0]
        label = dataset[i][-1]

        for t in range(250):
            gray_scale_feature_name = gray_scale_feature[t][0]
            mean_pixel_feature_name = mean_pixel_feature[t][0]
            extracting_edge_feature_name = extracting_edge_feature[t][0]
            hog_feature_name = hog_feature[t][0]

            if name == gray_scale_feature_name:
                gray_scale_feature[t][0] = label

            if name == mean_pixel_feature_name:
                mean_pixel_feature[t][0] = label

            if name == extracting_edge_feature_name:
                extracting_edge_feature[t][0] = label

            if name == hog_feature_name:
                hog_feature[t][0] = label

    np.save('gray_scale_feature.npy', gray_scale_feature)
    np.save('mean_pixel_feature.npy', mean_pixel_feature)
    np.save('extracting_edge_feature.npy', extracting_edge_feature)
    np.save('hog_feature.npy', hog_feature)


class TestDataPoint:
    def __init__(self):
        self.data = []
        self.total_data = 94

    def getData(self):
        data_file = open('test.csv', 'r')
        for index, data in enumerate(data_file):
            # skip first line
            if index:
                # split by , will cause a problem where the location also have , in them
                data = data.split(',')

                # the train data we have sometimes doesnt have all the value so we need to take care
                # of that by removing every other feature based on the index in the data
                # and what left is the location

                image_name = data[0]
                data.pop(0)

                gender = data[0]
                data.pop(0)

                age = data[0]
                data.pop(0)

                # we only need to care about the city
                location = data[-1]

                self.data.append((image_name, gender, age, location))

        self.data = np.array(self.data)
        np.save('test_data.npy', self.data)


if __name__ == "__main__":
    TRAIN_IMAGE_DIR = 'train'
    TEST_IMAGE_DIR = 'test'
    RESIZE_TRAIN_IMAGE_DIR = 'resized_train'
    RESIZE_TEST_IMAGE_DIR = 'resized_test'
    # make folder if its not existed
    try:
        os.mkdir(RESIZE_TRAIN_IMAGE_DIR)
    except FileExistsError:
        pass

    try:
        os.mkdir(RESIZE_TEST_IMAGE_DIR)
    except FileExistsError:
        pass

    """
        Get the basic information of our train data

        If this is the first time you run the code
        Please uncomment 'getTrainImageInfo()' to get the resize images
    """
    getImageInfo('train')
    print('Finished resizing train images.')

    """
        Get the basic information of our test data

        If this is the first time you run the code
        Please uncomment 'getTrainImageInfo()' to get the resize images
    """
    getImageInfo('test')
    print('Finished resizing test images.')

    """
        Extract image feature

        You can uncomment the next 2 line to recompute the feature np arrays. It will save the the new array feature,
        so be careful you might overwrite the old feature arrays.
    """
    train_images = ImagePreprocessor()
    train_images.featureExtraction()
    print('Finished extracting features from images.')

    """
        Visualize train data

        You can uncomment line train_data.getData() line to get the data again if you want
    """
    train_data = DataPoint()
    train_data.getData()
    train_data.visualizeFeature()
    print('Finished visualize the data and compute entropy.')

    """
        Update the feature label
    """
    updateLabelToFeature()
    print('Finished fixing label in feature arrays.')

    """
        Prepare test data
    """
    test_data = TestDataPoint()
    test_data.getData()
    print('Finished load test data.')