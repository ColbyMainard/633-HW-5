from keras import models
from numpy.core.defchararray import array

import keras
from hyperopt import hp, fmin, tpe

from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.models import model_from_json
import numpy as np
from numpy.lib.function_base import average

from sklearn.model_selection import StratifiedKFold

import machine_learning_models

x_data = machine_learning_models.load_image_directory("resized_train")
x_del, y_data = machine_learning_models.parse_csv_data("train.csv")
del x_del

# x ray only neural network
def hyperopt_space_xray_only():
	possible_layer_counts = [3,4,5,6,7,8]
	possible_compiler_optimizers = ['ada', 'rmsprop', 'sgd']
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

def find_optimal_parameters_xray_only(args):
	str_val = ""
	for argument in args:
		str_val += str(argument) + ", "
	print("Argument Values:", str_val)
	case, layer_count, optimizer, activator_fun, epoch_count, layer_type = args
	num_pixels = 600*600
	folds = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
	accuracies_list = []
	for train_index, test_index in folds.split(x_data, y_data):
		X_train, X_test = x_data[train_index], x_data[test_index] #x train/test split
		y_train, y_test = y_data[train_index], y_data[test_index] #y train/test split
		temp_model = Sequential()
		temp_model.add(Conv2D(32, kernel_size=3, activation=activator_fun, input_shape=(600,600,1))) #add input layer
		divisor_val = 1
		for idx in range(0, layer_count):
			temp_model.add(Conv2D(32, kernel_size=3, activation=activator_fun)) #convolutional layer
			if layer_type == Dense:
				temp_model.add(Dense(num_pixels // divisor_val, activation=activator_fun)) #intermediate_layer
			if layer_type == MaxPooling2D:
				temp_model.add(MaxPooling2D(pool_size=(3,3))) #intermediate_layer
			temp_model.add(Dropout(0.1))
			divisor_val *= 2
		Dense(2, activation='softmax')
		temp_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'], epochs=epoch_count, batch_size=200)
		temp_model.fit(X_train, y_train)
		accuracies_list.append(temp_model.history['accuracy'][-1])
	return 1 - average(accuracies_list)

if __name__ == "__main__":
	hyperopt_space = hyperopt_space_xray_only()
	best_parameters = fmin(find_optimal_parameters_xray_only, hyperopt_space, algo=tpe.suggest, max_evals=500)
	print("Best parameters:", best_parameters)