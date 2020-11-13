# this file will contain all machine learning models, including adaboost

#import the preprocessing

#import various machine learning libraries

from keras import models
import util_methods

import keras
from hyperopt import hp, fmin, tpe

from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.models import model_from_json

class CovidLearner():
	def __init__(self, training_data_files, test_data_files):
		self.training_files = training_data_files
		self.testing_files = test_data_files
	
	def save_model(self, filename):
		print("Save model to", filename)
		util_methods.raiseNotDefined()
	
	def load(self, filename):
		print("Load model from", filename)
		util_methods.raiseNotDefined()
	
	def train(self):
		print("Train model from data.")
		util_methods.raiseNotDefined()
	
	def predict(self):
		print("Make prediction from test data.")
		util_methods.raiseNotDefined()
	
	def optimize(self):
		print("Find optimal parameters.")
		util_methods.raiseNotDefined()

class CovidAdaBoostLearner(CovidLearner):
	#implement b.iv. here
	def __init__(self, training_data_files, test_data_files):
		self.training_files = training_data_files
		self.testing_files = test_data_files
	
	def save_model(self, filename):
		print("Save model to", filename)
		util_methods.raiseNotDefined()
	
	def load(self, filename):
		print("Load model from", filename)
		util_methods.raiseNotDefined()
	
	def train(self):
		print("Train model from data.")
		util_methods.raiseNotDefined()
	
	def predict(self):
		print("Make prediction from test data.")
		util_methods.raiseNotDefined()
	
	def optimize(self):
		print("Find optimal parameters.")
		util_methods.raiseNotDefined()

class CovidImageNeuralNetworkLearner(CovidLearner):
	#neural network processing of image-related data will be implemented here
	def __init__(self, training_data_files, test_data_files, intermediate_layer_type = Dense):
		self.training_files = training_data_files
		self.testing_files = test_data_files
		possible_layer_counts = [3,4,5,6]
		possible_compiler_optimizers = ['ada', 'rmsprop']
		activation_functions = ['relu', 'sigmoid']
		epoch_count = range(1,10) * 30
		cases_list = []
		case_number = 1
		self.non_conv_layer_type = intermediate_layer_type
		for layer_count in possible_layer_counts:
			for optimizer in possible_compiler_optimizers:
				for activators in activation_functions:
					for epoch_num in epoch_count:
						case_string = "case " + str(case_number)
						cases_list.append((case_string, layer_count, optimizer, activators, epoch_num, hp.uniform("dropout", 0, 1)))
		self.hyperopt_space = hp.choice('a', cases_list)
		self.model = None
	
	def save_model(self, json_file_name, h5_file_name):
		print("Save model json to", json_file_name, "and the weights will be stored at", h5_file_name)
		#https://machinelearningmastery.com/save-load-keras-deep-learning-models/
		model_json = self.model.to_json()
		with open(json_file_name, "w") as json_file:
			json_file.write(model_json)
		self.model.save_weights(h5_file_name)
		#util_methods.raiseNotDefined()
	
	def load(self, json_file_name, h5_file_name):
		print("Load model json from", json_file_name, "and the weights will be loaded from", h5_file_name)
		#https://machinelearningmastery.com/save-load-keras-deep-learning-models/
		json_file = open(json_file_name, 'r') #read structure of network
		loaded_model_json = json_file.read() #lad file into memory
		json_file.close()
		self.model = model_from_json(loaded_model_json)
		self.model.load_weights(h5_file_name)
		#util_methods.raiseNotDefined()
	
	def train(self):
		print("Train model from data.")
		util_methods.raiseNotDefined()
	
	def predict(self):
		print("Make prediction from test data.")
		util_methods.raiseNotDefined()

	def optimize(self):
		print("Find optimal parameters.")
		#http://hyperopt.github.io/hyperopt/
		def optimize_convolutional_network(args):
			case, layer_count, optimizer, activator, epoch_num, dropout_val = args
			self.model = models.Sequential()
			self.model.add(Conv2D(32, kernel_size=3, activation=activator, input_shape=(48,48,1)))#add convolutional input layer
			for i in range(0, layer_count):
				self.model.add()#add non-convolutional layer
				self.model.add()#add convolutional layer
				self.model.add()#add dropout layer
			#split training data into x data and y data, then perform 5 fold cross validation and get average loss across all folds
			for train_index, test_index in folds.split(x_data, y_data):
				X_train, X_test = x_data[train_index], x_data[test_index] #x train/test split
				y_train, y_test = y_data[train_index], y_data[test_index] #y train/test split
				self.model.compile(optimizer=optimizer, loss='binary_crossentropy')
				model_history = self.model.fit(x_train, to_categorical(y_train), epochs=10, batch_size=256,)
		best = fmin(optimize_convolutional_network, self.hyperopt_space, algo=tpe.suggest, max_evals=500)
		util_methods.raiseNotDefined()

class CovidPersonalDataLearner(CovidLearner):
	#processing of non-image related data will be done here
	def __init__(self, training_data_files, test_data_files):
		self.training_files = training_data_files
		self.testing_files = test_data_files
	
	def save_model(self, filename):
		print("Save model to", filename)
		util_methods.raiseNotDefined()
	
	def load(self, filename):
		print("Load model from", filename)
		util_methods.raiseNotDefined()
	
	def train(self):
		print("Train model from data.")
		util_methods.raiseNotDefined()
	
	def predict(self):
		print("Make prediction from test data.")
		util_methods.raiseNotDefined()
	
	def optimize(self):
		print("Find optimal parameters.")
		#http://hyperopt.github.io/hyperopt/
		util_methods.raiseNotDefined()

class CovidCombinationDataLearner():
	#meta learning agent that accepts multiple inputs regarding diagnoses and outputs a result based on their response
	def __init__(self, training_data_files, test_data_files):
		self.training_files = training_data_files
		self.testing_files = test_data_files
	
	def save_model(self, filename):
		print("Save model to", filename)
		util_methods.raiseNotDefined()
	
	def load(self, filename):
		print("Load model from", filename)
		util_methods.raiseNotDefined()
	
	def train(self):
		print("Train model from data.")
		util_methods.raiseNotDefined()
	
	def predict(self):
		print("Make prediction from test data.")
		util_methods.raiseNotDefined()
	
	def optimize(self):
		print("Find optimal parameters.")
		#http://hyperopt.github.io/hyperopt/
		util_methods.raiseNotDefined()

#functions as test harness stub for later use
if __name__ == "__main__":
	print("Testing various learning agents...")
	print("\tAdaBoost tests...")
	print("\tImage processing tests...")
	print("\tPatient info tests...")
	print("\tCombined image and patient info tests...")