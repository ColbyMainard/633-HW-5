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
from machine_learning_models import load_keras_model, parse_csv_data, save_keras_model

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
	y_expected = y_expected.reshape(50, 2)
	y_actual = y_actual.reshape(50, 2)
	num = 0.0
	denom = 50.0
	y_diff = (y_actual - y_expected)
	for point in y_diff:
		point_val = abs(point[0])
		if(point_val < 0.5):
			num += 1
		#print("Point:", point)
	return num / denom

best_hyper_parameter_tuple = ()
best_hyper_parameter_correctness = 0

def get_average_cross_validation_accuracy(args):
	try:
		conv_layer_count, dense_layer_count, compiler_optimizer, active_fun = args
		print("Testing parameters:", args)
		csv_x_data, csv_y_data = parse_csv_data("train.csv")
		feature = []
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
			classifier = [Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(1, activation='softmax'),]
		elif dense_layer_count == 3:
			classifier = [Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(2, activation='softmax'),]
		elif dense_layer_count == 4:
			classifier = [Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(2, activation='softmax'),]
		elif dense_layer_count == 5:
			classifier = [Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(2, activation='softmax'),]
		elif dense_layer_count == 6:
			classifier = [Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(2, activation='softmax'),]
		elif dense_layer_count == 6:
			classifier = [Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(2, activation='softmax'),]
		elif dense_layer_count == 7:
			classifier = [Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(2, activation='softmax'),]
		image_filenames = csv_x_data[:,0]
		accuracy_vals = []
		folds = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
		for train_index, test_index in folds.split(image_filenames, csv_y_data):
			raw_x_train_data, raw_x_test_data = image_filenames[train_index], image_filenames[test_index]
			x_train_data = format_images("resized_train", raw_x_train_data)
			x_train_data = x_train_data.reshape(200, 600, 600, 1)
		#print("Training images shape:", str(x_train_data.shape))
			x_test_data = format_images("resized_train", raw_x_test_data)
			x_test_data = x_test_data.reshape(50, 600, 600, 1)
		#print("Test images shape:", str(x_test_data.shape))
			y_train_data, y_test_data = csv_y_data[train_index], csv_y_data[test_index]
			model = Sequential(feature+classifier)
			model.compile(optimizer=compiler_optimizer, loss='binary_crossentropy',metrics=['accuracy'],)
		#model.fit(x_train_data, y_train_data,epochs=5,batch_size=5, verbose=0)
			model.fit(x_train_data, to_categorical(y_train_data), epochs=10, batch_size=5, verbose=0)
			y_expected = model.predict(x_test_data)
			accuracy_vals.append(accuracy_percent(y_expected, to_categorical(y_test_data)))
		print("Accuracy vals: ", accuracy_vals)
		correctness = average(accuracy_vals)
		print("Average percent correct:", correctness)
		global best_hyper_parameter_correctness
		global best_hyper_parameter_tuple
		if correctness > best_hyper_parameter_correctness:
			best_hyper_parameter_tuple = (conv_layer_count, dense_layer_count, compiler_optimizer, active_fun)
			best_hyper_parameter_correctness = correctness
		return 1 - correctness
	except:
		return 1

def optimize_hyperparameters():
	possible_conv_layer_counts = [3,4,5,6,7]
	possible_dense_layer_counts = [2,3,4,5,6,7]
	possible_compiler_optimizers = ['rmsprop', 'sgd', 'adam']
	activation_functions = ['relu', 'sigmoid']
	case_list = []
	case_num = 1
	for conv_layer_count in possible_conv_layer_counts:
		for dense_layer_count in possible_dense_layer_counts:
			for compiler_optimizer in possible_compiler_optimizers:
				for active_fun in activation_functions:
					case_string = "case " + str(case_num)
					case_num += 1
					case_list.append((conv_layer_count, dense_layer_count, compiler_optimizer, active_fun))
	feature_space = hp.choice('a', case_list)
	fmin(get_average_cross_validation_accuracy, feature_space, algo=tpe.suggest, max_evals=30)

def implement_optimum_model(parameter_tuple):
	csv_x_data, csv_y_data = parse_csv_data("train.csv")
	conv_layer_count, dense_layer_count, compiler_optimizer, active_fun = parameter_tuple
	feature = []
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
		classifier = [Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(1, activation='softmax'),]
	elif dense_layer_count == 3:
		classifier = [Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(2, activation='softmax'),]
	elif dense_layer_count == 4:
		classifier = [Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(2, activation='softmax'),]
	elif dense_layer_count == 5:
		classifier = [Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(2, activation='softmax'),]
	elif dense_layer_count == 6:
		classifier = [Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(2, activation='softmax'),]
	elif dense_layer_count == 6:
		classifier = [Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(2, activation='softmax'),]
	elif dense_layer_count == 7:
		classifier = [Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(32, activation=active_fun), Dropout(0.1), Dense(2, activation='softmax'),]
	raw_x_data = csv_x_data[:,0]
	raw_x_data =  format_images("resized_train", raw_x_data)
	x_data = raw_x_data.reshape(250, 600, 600, 1)
	y_data = csv_y_data
	model = Sequential(feature+classifier)
	model.compile(optimizer=compiler_optimizer, loss='binary_crossentropy',metrics=['accuracy'],)
	model_history = model.fit(x_data, to_categorical(y_data), epochs=12, batch_size=5)
	print("Accuracy history:", model_history.history['accuracy'])
	save_keras_model(model, "optimum_xray_model.json","optimum_xray_model.h5")

if __name__ == "__main__":
	#optimize_hyperparameters()
	#implement_optimum_model(best_hyper_parameter_tuple)
	implement_optimum_model((3, 4, 'rmsprop', 'relu'))
	model = load_keras_model("optimum_xray_model.json","optimum_xray_model.h5")
	print("Model summary:")
	print(model.summary())