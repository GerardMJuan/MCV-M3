__author__ = "Sergi Sancho, Adriana Fernandez, Eric Lopez y Gerard Marti"
__credits__ = ['Sergi Sancho', 'Adriana Fernandez', 'Eric Lopez', 'Gerard Marti']
__license__ = "GPL"
__version__ = "1.0"

import os
from Tools import nms
from Tools import feature_extractor
import pickle
import numpy as np
from skimage import io
from skimage.util import pad
from skimage.transform import pyramid_gaussian
from skimage.util.shape import view_as_windows
import skimage.util as util
import math
import Config as cfg
import xgboost as xgb
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import collections
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cfg.data))
    plt.xticks(tick_marks, cfg.data, rotation=45)
    plt.yticks(tick_marks, cfg.data)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    res = cfg.resultsFolder + cfg.model + '_' + cfg.modelFeatures + '_confusion_matrix.jpg'
    plt.savefig(res)
    print 'Confusion matrix stored in: .../' + res 
	
def evaluate_results(gt_y,pred_y):
    #Store the predictions to a vector
    predictions = []
    for idx, val in enumerate(pred_y):
        predictions.append(val)
    
    # Confusion matrix
    labels = cfg.data
    cm = confusion_matrix(gt_y, predictions)
    print('Confusion matrix, without normalization')
    print(cm)	
    
    # Accuracy
    acc = accuracy_score(gt_y, predictions)
    print 'Accuracy: ' + str(acc)
    
    # F1 Measure
    fm = f1_score(gt_y, predictions, average=None)
    mfm = np.mean(fm)
    print 'Av. F1 Measure: ' + str(mfm)
    
    # Precision
    precision = precision_score(gt_y, predictions, average=None) 
    mprecision = np.mean(precision)
    print 'Av. Precision: ' + str(mprecision)
    
    # Sensitivity, Recall, TPR
    recall = recall_score(gt_y, predictions, average=None)
    mrecall = np.mean(recall)
    print 'Av. Sensitivity: ' + str(mrecall)    
    
    # Plot the confusion matrix
    plt.figure()
    plot_confusion_matrix(cm)
    
    '''
    #Save the prediction
    res = cfg.resultsFolder + '/' + 'Prediction_' + cfg.modelFeatures + '.txt'
    np.savetxt(
    res,   					# file name
    pred_y,                 # array to save
    fmt='%.2f',             # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character  
    comments='# ')          # character to use for comments
	'''
	
###################################
#WARNING, TO DO IN SLIDING WINDOW #
###################################
#1. Change the size of the GT in each pyramid iteration
#2. If there are more that one window that detect a signal, overlap them

def testImage_slidingwindow(filename, applyNMS=True):
    #Load the model
    bst = xgb.Booster({'nthread':cfg.xgParam['nthread']})
    bst.load_model(cfg.modelPath)
    
    #Load the current image
    imagePath = cfg.testFolderPath + '/' + filename
    image = io.imread(imagePath, as_grey=True)
    image = util.img_as_ubyte(image) #Read the image as bytes (pixels with values 0-255)
	
	#Load the gt
    dir_gt = cfg.annotationsFolderPath + '/' + filename
    dir_gt = dir_gt.replace('jpg','txt') # png or jpg
    with open(dir_gt,'r') as fp:
        for line in fp:
            gt_y1, gt_x1, gt_y2, gt_x2, gt_signal = line.split(' ', 4 )
            gt_y1 = float(gt_y1)
            gt_x1 = float(gt_x1)
            gt_y2 = float(gt_y2)
            gt_x2 = float(gt_x2)
            break
            
    #Scan the image
    iteration = 0
    rows, cols = image.shape
    pyramid = tuple(pyramid_gaussian(image, downscale=cfg.downScaleFactor))
    scale = 0
    boxes = None
    scores = None
    
    #Test data set
    X = []
    y = []
    
    for p in pyramid[0:]:
        #We now have the subsampled image in p      
        print p.shape     
		
        #Add padding to the image, using reflection to avoid border effects
        if cfg.padding > 0:
            p = pad(p,cfg.padding,'reflect')
        try:
            views = view_as_windows(p, cfg.window_shape, step=cfg.window_step)
        except ValueError:
            #block shape is bigger than image
            break

        num_rows, num_cols, width, height = views.shape
        for row in range(0, num_rows):
            for col in range(0, num_cols):
                iteration = iteration + 1
                
			    #Get current window
                subImage = views[row, col]

                #Window parameters
                hh, ww = cfg.window_shape
                xx1 = int(col*cfg.window_step - cfg.padding + cfg.window_margin)
                yy1 = int(row*cfg.window_step - cfg.padding + cfg.window_margin)
                xx2 = int(xx1 + (ww - 2*cfg.window_margin))
                yy2 = int(yy1 + (hh - 2*cfg.window_margin))
          
                print '--------------------[ window, it: '+ str(iteration) +' ]--------------------'
                gt_h = gt_y2-gt_y1;
                gt_w = gt_x2-gt_x1;
				
                feats = feature_extractor.extractFeatures(subImage) # Extract features
                if len(feats) > cfg.num_features:
                    print 'Warning: feats length '+str(len(feats)) + ' is resized to ' + str(cfg.num_features)
                    feats = feats[:cfg.num_features] #The length of the feats must be the same as the training data
                X.append(feats)
                #Apriori knowledge: if the current window overlaps the GT data.
                if (not(gt_x1 > xx1+ww or gt_x1+gt_w < xx1 or gt_y1 > yy1+hh or gt_y1+gt_h < yy1)):
                    print '=====> overlapping <====='
                    if gt_signal in cfg.Cuadradas: #Square
                        y.append(2.00)
                        #X.append(feats)
                    elif gt_signal in cfg.Triangulares: #Triangle
                        y.append(3.00)
                        #X.append(feats)
                    else: #Circle
                        y.append(1.00)
                        #X.append(feats)
                    bbox = (xx1, yy1, xx2, yy2)
                    if boxes is not None:
                        boxes = np.vstack((bbox, boxes))
                    else:
                        boxes = np.array([bbox])
                else: #Background
                    y.append(0.00)
                iteration = iteration + 1
        scale += 1
        scaleMult = math.pow(cfg.downScaleFactor, scale)
        break
    
    #Save the test data set to .data file in Data folder.
    np.savetxt(
    cfg.tDatasetPath,   	# file name
    np.c_[X,y],             # array to save
    fmt='%.2f',             # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character  
    comments='# ')          # character to use for comments
    
    #Compute the predictions
    xg_test = xgb.DMatrix(X, label=y)
    decision_func = bst.predict(xg_test);
    evaluate_results(y,decision_func) # Evaluate the results
    return boxes
    
def testFolder_slidingwindow(inputfolder, outputfolder, applyNMS=True):
    fileList = os.listdir(inputfolder)
    imagesList = filter(lambda element: '.jpg' in element, fileList)
    print ('Start processing '+inputfolder)
    for filename in imagesList:

        imagepath = inputfolder + '/' + filename
        print ('Processing '+imagepath)

        #Test the current image
        bboxes = testImage_slidingwindow(filename, applyNMS=applyNMS)
        
        #Store the result in a dictionary
        result = dict()
        result['imagepath'] = imagepath
        result['bboxes'] = bboxes
        #result['confusion_matrix'] = confusion_matrix

        #Save the features to a file using pickle
        outputFile = open(outputfolder+'/'+filename+'.results', "wb")
        pickle.dump(result, outputFile)
        outputFile.close()
        
def testFolder_perImage():
    #Load the model
    bst = xgb.Booster({'nthread':cfg.xgParam['nthread']})
    bst.load_model(cfg.modelPath)
    
    #Initialize the testing data
    X = np.zeros(shape=(cfg.num_train_images,cfg.num_features))
    y = np.zeros(shape=(cfg.num_train_images,1))
    
    extension = '.jpg'
    number_of_images = 0
    
    for index_label, name_label in enumerate(cfg.test_perImage_folders): # For each item...
        imagesPath = cfg.test_perImage_path + '/' + name_label # Each label corresponds to a folder
        fileList = os.listdir(imagesPath) # List all files
        imagesList = filter(lambda element: extension in element, fileList) # Select only the ones that ends with the desired extension
        for filename in imagesList:
            current_imagePath = imagesPath + '/' + filename   
            print ('Processing '+current_imagePath)
            image = io.imread(current_imagePath, as_grey=True)
            image = util.img_as_ubyte(image) # Read the image as bytes (pixels with values 0-255)
            X[number_of_images] = feature_extractor.extractFeatures(image) # Extract the features
            y[number_of_images] = index_label # Assign the label at the end of X when saving the data set
            number_of_images = number_of_images + 1         
            
    #Save the test data set to .data file in Data folder.
    np.savetxt(
    cfg.test_perImage_dataset,   		# file name
    np.c_[X,y],                         # array to save
    fmt='%.2f',                         # formatting, 2 digits in this case
    delimiter=',',                      # column delimiter
    newline='\n',                       # new line character  
    comments='# ')                      # character to use for comments
    
    #Compute the predictions
    xg_test = xgb.DMatrix(X, label=y)
    decision_func = bst.predict(xg_test);

    evaluate_results(y,decision_func)

def testFolder_colorenh_ccl():
    #Load the model
    bst = xgb.Booster({'nthread':cfg.xgParam['nthread']})
    bst.load_model(cfg.modelPath)
    
    #Initialize the testing data
    X = np.zeros(shape=(cfg.num_train_images,cfg.num_features))
    y = np.zeros(shape=(cfg.num_train_images,1))
    
	#Output structure
    boxes = None
    img_structure = None
    candidates = dict()
	
    extension = '.jpg'
    number_of_images = 0
    
    fileList = os.listdir(cfg.test_omatlab_folder_path) # List all files
    imagesList = filter(lambda element: extension in element, fileList) # Select only the ones that ends with the desired extension
    for filename in imagesList:
        current_imagePath = cfg.test_omatlab_folder_path + '/' + filename
        print ('Processing '+current_imagePath)
                
        img_name = filename.rsplit('_', 1)[0] # The name of the original image	        
		
        # Fill X
        image = io.imread(current_imagePath, as_grey=True)
        image = util.img_as_ubyte(image) # Read the image as bytes (pixels with values 0-255)
        X[number_of_images] = feature_extractor.extractFeatures(image) # Extract the features
		
		# Fill Y
        local_dir_gt = current_imagePath
        local_dir_gt = local_dir_gt.replace('jpg','txt') # png or jpg
        with open(local_dir_gt,'r') as fp:
            for line in fp:
                local_gt_y1, local_gt_x1, local_gt_y2, local_gt_x2, local_gt_sign = line.split(' ', 4 )
                local_gt_y1 = float(local_gt_y1)
                local_gt_x1 = float(local_gt_x1)
                local_gt_y2 = float(local_gt_y2)
                local_gt_x2 = float(local_gt_x2)
                local_gt_sign = local_gt_sign.replace('\n','')
                bbox = (img_name, cfg.data.index(local_gt_sign), local_gt_x1, local_gt_y1, local_gt_x2, local_gt_y2)
                if boxes is not None:
                    boxes = np.vstack((bbox, boxes))
                else:
                    boxes = np.array([bbox])
            fp.close
            y[number_of_images] = cfg.data.index(local_gt_sign) # Suponiendo que sean las mismas clases que data
            print str(local_gt_y1) + ', ' + str(local_gt_x1) + ', ' + str(local_gt_y2) + ', ' + str(local_gt_x2) + ', ' + local_gt_sign
        number_of_images = number_of_images + 1
    
    #Save the test data set to .data file in Data folder.
    np.savetxt(
    cfg.ce_ccl_test_dataset,   			# file name
    np.c_[X,y],                         # array to save
    fmt='%.2f',                         # formatting, 2 digits in this case
    delimiter=',',                      # column delimiter
    newline='\n',                       # new line character  
    comments='# ')                      # character to use for comments
    
    #Compute the predictions
    xg_test = xgb.DMatrix(X, label=y)
    decision_func = bst.predict(xg_test);
    
	#Return those bounding boxes that its labels predicted are not background.
    candidates['bboxes'] = boxes # Store the bounding boxes
    candidates['prediction'] = decision_func[::-1] # Store the reversed array of predictions

    outputFile = open(cfg.resultsFolder+'/Ce_ccl.results', "wb")
    pickle.dump(candidates, outputFile)
    outputFile.close()

    #print candidates['bboxes']
    #print decision_func
	
	#Evaluate the results
    evaluate_results(y,decision_func)
