# this file will contain all machine learning models, including adaboost

#import the preprocessing

#import various machine learning libraries

import os
import cv2

from keras import models
from numpy.core.defchararray import array
import util_methods

import keras
from hyperopt import hp, fmin, tpe

from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.models import model_from_json
import numpy as np

from natsort import natsorted

import subprocess

from sklearn.model_selection import StratifiedKFold

import pandas as pd

from adaboost import *

#returns a list of grayscale images
def load_image_directory(directory_name):
	image_list = []
	#for each filename in the directory
	for file_name in os.listdir(directory_name):
		#get path to file
		complete_path = os.path.join(directory_name, file_name)
		#load the image in grayscale, and append it tp the list
		image_list.append(cv2.imread(complete_path, 0))
	return image_list

#used to load and parse the test.csv and train.csv
#returns a tuple containing two numpy arrays in the form (categories, label)
def parse_csv_data(csv_filename, has_label=True):#has label will be true for training data, false for testing data
	raw_data = pd.read_csv(csv_filename)
	parsed_x_data = []
	parsed_y_data = []
	col_1 = raw_data["filename"]
	col_2 = raw_data["gender"]
	col_3 = raw_data["age"]
	col_4 = raw_data["location"]
	x_data = []
	y_data = []
	for idx in range(0, len(col_1)):
		x_data.append([col_1[idx], col_2[idx], col_3[idx], col_4[idx]])
		if has_label:
			y_data.append(to_categorical(raw_data["covid(label)"][idx]))
	return (np.array(x_data), np.array(y_data))



#for neural networks
def save_keras_model(model, json_file_name, h5_file_name):
	model_json = model.to_json()
	with open(json_file_name, "w") as json_file:
		json_file.write(model_json)
	model.save_weights(h5_file_name)

def load_keras_model(json_file_name, h5_file_name):
	#https://machinelearningmastery.com/save-load-keras-deep-learning-models/
	json_file = open(json_file_name, 'r') #read structure of network
	loaded_model_json = json_file.read() #lad file into memory
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights(h5_file_name)
	return model

#functions as test harness stub for later use
if __name__ == "__main__":
	print("Loading preprocessed images...")
	training_images = load_image_directory("train")
	print("\tTraining images loaded...")
	test_images = load_image_directory("test")
	print("\tTest images loaded...")
	print("Loading csv data...")
	training_data_x, training_data_y = parse_csv_data("train.csv")
	testing_data_x, testing_data_y = parse_csv_data("test.csv", False)
	print("Testing various learning agents...")
	print("\tAdaBoost tests...")	

	# params = {}
	# params['n_estimators'] = 100
	# model = adaboost_classifier(training_images, training_data_y, params)
	# print(adaboost_score(model, test_images, testing_data_y))

	print("\tImage processing tests...")
	print("\tPatient info tests...")
	print("\tCombined image and patient info tests...")
