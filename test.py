__author__ = "Sergi Sancho, Adriana Fernandez, Eric Lopez y Gerard Marti"
__credits__ = ['Sergi Sancho', 'Adriana Fernandez', 'Eric Lopez', 'Gerard Marti']
__license__ = "GPL"
__version__ = "1.0"

from Tools import detector
from MatlabCode import detect_signals
import Config as cfg
import pickle
import os
def run():

    # Create the results directory if it doesn't exist
    if not os.path.exists(cfg.resultsFolder):
        os.makedirs(cfg.resultsFolder)
		
    # Compute the test per image or by means of sliding window method
    if cfg.exp_methodology == 2:
        #Get the crops of the test images using MATLAB
        detect_signals.run()
        #Classify the images
        detector.testFolder_colorenh_ccl()	
    elif cfg.exp_methodology == 1:
        detector.testFolder_perImage()
    else:
        detector.testFolder_slidingwindow(cfg.testFolderPath,cfg.resultsFolder,applyNMS=False)    

if __name__ == '__main__':
    run()

