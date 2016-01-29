__author__ = "Sergi Sancho, Adriana Fernandez, Eric Lopez y Gerard Marti"
__credits__ = ['Sergi Sancho', 'Adriana Fernandez', 'Eric Lopez', 'Gerard Marti']
__license__ = "GPL"
__version__ = "1.0"

from Tools import feature_extractor

from skimage import io
import os
import Config as cfg
from skimage import util
import numpy as np
import pickle

def run():
    print 'Extracting features from images in '+cfg.trainingFolderPath
    extractAndStoreFeatures(cfg.trainingFolderPath, cfg.FeaturesInPath, cfg.pDatasetPath)

def extractAndStoreFeatures(inputFolder, items, outputFolder):
	extension = '.jpg'
	X = np.zeros(shape=(cfg.num_train_images,cfg.num_features))
	y = np.zeros(shape=(cfg.num_train_images,1))
	number_of_images = 0
	for index_label, name_label in enumerate(items): # For each item...
		imagesPath = inputFolder + '/' + name_label # Each label corresponds to a folder
		fileList = os.listdir(imagesPath) # List all files
		imagesList = filter(lambda element: extension in element, fileList) # Select only the ones that ends with the desired extension
		for filename in imagesList:
			current_imagePath = imagesPath + '/' + filename
			print 'Extracting features for ' + current_imagePath
			image = io.imread(current_imagePath, as_grey=True)
			image = util.img_as_ubyte(image) # Read the image as bytes (pixels with values 0-255)
			X[number_of_images] = feature_extractor.extractFeatures(image) # Extract the features
			y[number_of_images] = index_label # Assign the label at the end of X when saving the data set
			number_of_images = number_of_images + 1
            
	#Save the data set to .data file in Data folder.
	np.savetxt(
	outputFolder,   		# file name
	np.c_[X,y],             # array to save
	fmt='%.2f',             # formatting, 2 digits in this case
	delimiter=',',          # column delimiter
	newline='\n',           # new line character  
	comments='# ')          # character to use for comments

if __name__ == '__main__':
    run()

