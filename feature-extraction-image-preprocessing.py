# image-preprocessor.py
# This file will contain all necessary methods and data to implement a.-b.iii.
# This translates to image preprocessing, visual feature extraction, feature exploration, and feature selection
# All the above functionality should be made available in methods/classes as to allow for easy access in other files

class ImagePreprocessor():
	#image preprocessing implementation goes here
	def __init__(self,directory_name):
		self.data_directory = directory_name #directory containing data

	def load_image(self, filename):
		print("Returns an unprocessed image.")
	
	def process_image(image):
		print("Returns a processed image.")
    
class VisualFeatureExtractor():
	#visual feature extraction implementation goes here
	def __init__(self):
		self.features_list = []
		self.image_prepper = ImagePreprocessor()
		print("Implement later.")
    
class FeatureExplorer():
	#feature exploration goes here
	def __init__(self):
		self.feature_extractor = VisualFeatureExtractor()
		print("Implement later")
	
	def explore_features(self):
		print("Do cursory examination on each feature found in VisualFeatureExtractor.")

class FeatureSelector():
	#feature selection goes here
	def __init__(self):
		self.feature_explorer = FeatureExplorer()
		print("Implement later")
	
	def get_most_important_features():
		print("Use this method to get most important features, as analyzed in feature explorer.")

if __name__ == "__main__":
	#add tests for each class below
	print("Image preprocessing tests...")
	print("Visual feature extraction tests...")
	print("Feature exploration tests...")
	print("Feature selection tests...")
