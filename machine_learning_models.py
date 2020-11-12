# this file will contain all machine learning models, including adaboost

#import the preprocessing

#import various machine learning libraries

import util_methods

import keras
import hyperopt

from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Flatten



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

def hyperopt_image_neural_network():

class CovidImageNeuralNetworkLearner(CovidLearner):
	#neural network processing of image-related data will be implemented here
	def __init__(self, training_data_files, test_data_files):
		self.training_files = training_data_files
		self.testing_files = test_data_files
	
	def save_model(self, filename):
		print("Save model to", filename)
		#https://machinelearningmastery.com/save-load-keras-deep-learning-models/
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