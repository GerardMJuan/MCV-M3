__author__ = "Sergi Sancho, Adriana Fernandez, Eric Lopez y Gerard Marti"
__credits__ = ['Sergi Sancho', 'Adriana Fernandez', 'Eric Lopez', 'Gerard Marti']
__license__ = "GPL"
__version__ = "1.0"

import extract_features
import train
import test
import draw_results

extract_features.run() # Extracts the features for all the images
train.run() # Train the classifier
test.run() # Test a whole folder
draw_results.run() # Draw the detected boxes in the images