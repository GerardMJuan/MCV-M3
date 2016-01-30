__author__ = "Sergi Sancho, Adriana Fernandez, Eric Lopez y Gerard Marti"
__credits__ = ['Sergi Sancho', 'Adriana Fernandez', 'Eric Lopez', 'Gerard Marti']
__license__ = "GPL"
__version__ = "1.0"

import extract_features
import train
import test
import draw_results
from MatlabCode import detect_signals

#DETECTION

#Color segmentation
detect_signals.run()

#CLASSIFICATION
extract_features.run() # Extracts the features for all the images to train the classifier
train.run() # Train the classifier

#TESTING AND EVALUATION
test.run() # Test a whole folder
draw_results.run() # Draw the detected boxes in the images