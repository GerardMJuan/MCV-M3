__author__ = "Sergi Sancho, Adriana Fernandez, Eric Lopez y Gerard Marti"
__credits__ = ['Sergi Sancho', 'Adriana Fernandez', 'Eric Lopez', 'Gerard Marti']
__license__ = "GPL"
__version__ = "1.0"

import numpy as np
import Config as cfg
import pickle
import xgboost as xgb
import os

def run():
	train_data = np.loadtxt(cfg.pDatasetPath, delimiter=',') # Load the data set
	sz = train_data.shape
	train_X = train_data[:,0:(cfg.num_features)] # Assign the features to train_X
	train_Y = train_data[:, cfg.num_features] # Assign the labels to train_Y
	xg_train = xgb.DMatrix(train_X, label=train_Y)
	print 'Building the data set'
	bst = xgb.train(cfg.xgParam, xg_train, cfg.xg_num_round) # Start training
	bst.save_model(cfg.modelPath) # Save the model
	print 'Data set stored in '+cfg.modelPath
	
if __name__ == '__main__':
    run()