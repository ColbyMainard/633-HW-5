from json import load
from threading import current_thread
from keras import models
from numpy.core.defchararray import array

from hyperopt import hp, fmin, tpe

from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.models import model_from_json
import numpy as np
from numpy.lib.function_base import average

from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.engine import sequential

import machine_learning_models
from machine_learning_models import parse_csv_data

import os
import cv2

# x ray only neural network
def hyperopt_space_xray_only():
	possible_layer_counts = [3,4,5,6,7,8]
	possible_compiler_optimizers = ['rmsprop', 'sgd']
	activation_functions = ['relu', 'sigmoid']
	epoch_count = range(1,10)
	for count in epoch_count:
		count *= 30
	cases_list = []
	case_number = 1
	non_conv_layer_types = [Dense, MaxPooling2D]
	for layer_count in possible_layer_counts:
		for optimizer in possible_compiler_optimizers:
			for activators in activation_functions:
				for epoch_num in epoch_count:
					for layer_type in non_conv_layer_types:
						case_string = "case " + str(case_number)
						cases_list.append((case_string, layer_count, optimizer, activators, epoch_num, layer_type))
	return hp.choice('a', cases_list)

def percent_predictions_correct(y_expected, y_actual):
	num = 0.0
	denom = 0.0
	for idx in range(0, len(y_expected)):
		if y_expected[idx] == y_actual[idx]:
			num += 1
		denom += 1
	return (num / denom)

def format_images(directory_name, image_filename_list):
	formatted_images = []
	for image_filename in image_filename_list:
		image_filename = os.path.join(directory_name, image_filename)
		loaded_image = cv2.imread(image_filename, 0)
		loaded_image = np.array(loaded_image)
		loaded_image = (loaded_image / 255) - .5
		loaded_image.reshape(600,600,1)
		formatted_images.append(loaded_image)
	formatted_images = np.array(formatted_images)
	return formatted_images

def get_cross_validation_accuracy(layer_count, optimizer, activator_fun, layer_type, epoch_count):
	model = Sequential()
	num_pixels = 600*600
	denom_val = 1
	model.add(Conv2D(32, kernel_size=3, activation=activator_fun, input_shape=(600,600,1,1)))
	for idx in range (0, layer_count):
		model.add(Conv2D(32, kernel_size=(1,1), activation=activator_fun))
		if layer_type == Dense:
			model.add(Dense(num_pixels // denom_val, activation=activator_fun))
		if layer_type == MaxPooling2D:
			model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.1))
		denom_val *= 2
	model.add(Dense(2, activation='softmax'))
	model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	csv_x_data, csv_y_data = parse_csv_data("train.csv")
	accuracy_list = []
	image_filenames = csv_x_data[:,0]
	folds = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
	for train_index, test_index in folds.split(image_filenames, csv_y_data):
		raw_x_train_data, raw_x_test_data = image_filenames[train_index], image_filenames[test_index]
		x_train_data = format_images("resized_train", raw_x_train_data)
		#x_train_data = np.array(x_train_data)
		x_train_data = x_train_data.reshape(200, 600, 600)
		print("Training images shape:", str(x_train_data.shape))
		x_test_data = format_images("resized_train", raw_x_test_data)
		#x_test_data = np.array(x_test_data)
		x_test_data = x_test_data.reshape(50, 600, 600)
		print("Test images shape:", str(x_test_data.shape))
		y_train_data, y_test_data = to_categorical(csv_y_data[train_index], num_classes=2), to_categorical(csv_y_data[test_index], num_classes=2)
		model_history = model.fit(x_train_data, y_train_data, epochs=epoch_count, batch_size=200)
		predictions = model.predict(x_test_data)
		accuracy_list.append(percent_predictions_correct(predictions, y_test_data))
	return 1 - average(accuracy_list)

def find_optimal_parameters_xray_only(args):
	case, layer_count, optimizer, activator_fun, epoch_count, layer_type = args
	del case
	return 1 - get_cross_validation_accuracy(layer_count, optimizer, activator_fun, layer_type, epoch_count)

if __name__ == "__main__":
	hyperopt_space = hyperopt_space_xray_only()
	best_parameters = fmin(find_optimal_parameters_xray_only, hyperopt_space, algo=tpe.suggest, max_evals=5000)
	print("Best parameters:", best_parameters)
