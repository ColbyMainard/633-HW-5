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
#ADD ADA CLASSIFIER ITEMS HERE

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

# x ray only neural network
def hyperopt_space_xray_only():
	possible_layer_counts = [3,4,5,6]
	possible_compiler_optimizers = ['ada', 'rmsprop']
	activation_functions = ['relu', 'sigmoid']
	epoch_count = range(1,10) * 30
	cases_list = []
	case_number = 1
	non_conv_layer_types = [Dense, MaxPooling2D]
	for layer_count in possible_layer_counts:
		for optimizer in possible_compiler_optimizers:
			for activators in activation_functions:
				for epoch_num in epoch_count:
					for layer_type in non_conv_layer_types:
						case_string = "case " + str(case_number)
						cases_list.append((case_string, layer_count, optimizer, activators, epoch_num, layer_type, hp.uniform("dropout", 0, 1)))
	return hp.choice('a', cases_list)

def find_optimal_parameters_xray_only(x_data, y_data):
	folds = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
	for train_index, test_index in folds.split(x_data, y_data):
		X_train, X_test = x_data[train_index], x_data[test_index] #x train/test split
		y_train, y_test = y_data[train_index], y_data[test_index] #y train/test split

#functions as test harness stub for later use
if __name__ == "__main__":
	print("Loading preprocessed images...")
	training_images = load_image_directory("resized_train")
	print("\tTraining images loaded...")
	test_images = load_image_directory("resized_test")
	print("\tTest images loaded...")
	print("Loading csv data...")
	training_data_x, training_data_y = parse_csv_data("train.csv")
	testing_data_x, testing_data_y = parse_csv_data("test.csv", False)
	print("Testing various learning agents...")
	print("\tAdaBoost tests...")
	print("\tImage processing tests...")
	print("\tPatient info tests...")
	print("\tCombined image and patient info tests...")