# this file will contain all machine learning models, including adaboost

#import the preprocessing

#import various machine learning libraries

import util_methods

class CovidLearner():
	#add common features to all learners, like save, load, train, predict
	def __init__(self, training_data_files, test_data_files):
		self.training_files = training_data_files
		self.testing_files = test_data_files
		util_methods.raiseNotDefined()
	
	def save_model(self, filename):
		print("Save model to", filename)

class CovidAdaBoostLearner(CovidLearner):
	#implement b.iv. here
	def __init__(self):
		print("Override methods as necessary.")
		util_methods.raiseNotDefined()

class CovidImageNeuralNetworkLearner(CovidLearner):
	#neural network processing of image-related data will be implemented here
	def __init__(self):
		print("Override methods as necessary.")
		util_methods.raiseNotDefined()

class CovidPersonalDataLearner(CovidLearner):
	#processing of non-image related data will be done here
	def __init__(self):
		print("Override methods as necessary.")
		util_methods.raiseNotDefined()

class CovidCombinationDataLearner():
	#meta learning agent that accepts multiple inputs regarding diagnoses and outputs a result based on their response
	def __init__(self):
		print("Override methods as necessary.")
		util_methods.raiseNotDefined()

#functions as test harness stub for later use
if __name__ == "__main__":
	print("Testing various learning agents...")
	print("\tAdaBoost tests...")
	print("\tImage processing tests...")
	print("\tPatient info tests...")
	print("\tCombined image and patient info tests...")