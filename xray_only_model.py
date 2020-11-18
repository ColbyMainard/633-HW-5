from json import load
from threading import current_thread
from keras import models
from numpy.core.defchararray import array

from hyperopt import hp, fmin, tpe

from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import model_from_json
import numpy as np
from numpy.lib.function_base import average

from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.engine import sequential

import machine_learning_models
from machine_learning_models import parse_csv_data, save_keras_model

import os
import cv2

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

def accuracy_percent(y_expected, y_actual):
	num = 0.0
	denom = 0.0
	for idx in len(y_expected):
		if(y_expected[idx] == y_actual[idx]):
			num += 1
		denom += 1
	return num / denom

def get_average_cross_validation_accuracy(args):
	case_string, conv_layer_count, dense_layer_count, compiler_optimizer, active_fun = args
	csv_x_data, csv_y_data = parse_csv_data("train.csv")
	feature = []
	print("Case:", case_string)
	if(conv_layer_count == 3):
		feature = [Conv2D(4, kernel_size=3, activation=active_fun, input_shape=(600,600,1)), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Flatten(),]
	elif(conv_layer_count == 4):
		feature = [Conv2D(4, kernel_size=3, activation=active_fun, input_shape=(600,600,1)), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Flatten(),]
	elif(conv_layer_count == 5):
		feature = [Conv2D(4, kernel_size=3, activation=active_fun, input_shape=(600,600,1)), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Flatten(),]
	elif(conv_layer_count == 6):
		feature = [Conv2D(4, kernel_size=3, activation=active_fun, input_shape=(600,600,1)), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Flatten(),]
	elif(conv_layer_count == 7):
		feature = [Conv2D(4, kernel_size=3, activation=active_fun, input_shape=(600,600,1)), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Conv2D(4, kernel_size=3, activation=active_fun), MaxPooling2D(pool_size=2), Flatten(),]
	classifier = []
	if dense_layer_count == 2:
		classifier = [Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(2, activation='softmax'),]
	elif dense_layer_count == 3:
		classifier = [Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(2, activation='softmax'),]
	elif dense_layer_count == 4:
		classifier = [Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(2, activation='softmax'),]
	elif dense_layer_count == 5:
		classifier = [Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(2, activation='softmax'),]
	elif dense_layer_count == 6:
		classifier = [Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(2, activation='softmax'),]
	elif dense_layer_count == 6:
		classifier = [Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(32, activation=active_fun), Dense(2, activation='softmax'),]
	image_filenames = csv_x_data[:,0]
	accuracy_vals = []
	folds = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
	for train_index, test_index in folds.split(image_filenames, csv_y_data):
		raw_x_train_data, raw_x_test_data = image_filenames[train_index], image_filenames[test_index]
		x_train_data = format_images("resized_train", raw_x_train_data)
		x_train_data = x_train_data.reshape(200, 600, 600, 1)
		print("Training images shape:", str(x_train_data.shape))
		x_test_data = format_images("resized_train", raw_x_test_data)
		x_test_data = x_test_data.reshape(50, 600, 600, 1)
		print("Test images shape:", str(x_test_data.shape))
		y_train_data, y_test_data = to_categorical(csv_y_data[train_index], num_classes=2), to_categorical(csv_y_data[test_index], num_classes=2)
		model = Sequential(feature+classifier)
		model.compile(optimizer=compiler_optimizer, loss='categorical_crossentropy',metrics=['accuracy'],)
		model.fit(x_train_data, y_train_data,epochs=20,batch_size=25)
		accuracy_vals.append(accuracy_percent(model.predict(x_test_data), y_test_data))
	return 1 - average(accuracy_vals)

def optimize_hyperparameters():
	possible_conv_layer_counts = [3,4,5,6,7]
	possible_dense_layer_counts = [2,3,4,5,6,7]
	possible_compiler_optimizers = ['rmsprop', 'sgd']
	activation_functions = ['relu', 'sigmoid']
	case_list = []
	case_num = 1
	for conv_layer_count in possible_conv_layer_counts:
		for dense_layer_count in possible_dense_layer_counts:
			for compiler_optimizer in possible_compiler_optimizers:
				for active_fun in activation_functions:
					case_string = "case " + str(case_num)
					case_num += 1
					case_list.append((case_string, conv_layer_count, dense_layer_count, compiler_optimizer, active_fun))
	feature_space = hp.choice('a', case_list)
	best_parameters = fmin(get_average_cross_validation_accuracy, feature_space, algo=tpe.suggest, max_evals=250)
	print("Best parameters:", best_parameters)
	
if __name__ == "__main__":
        optimize_hyperparameters()
