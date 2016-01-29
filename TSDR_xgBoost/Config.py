__author__ = "Sergi Sancho, Adriana Fernandez, Eric Lopez y Gerard Marti"
__credits__ = ['Sergi Sancho', 'Adriana Fernandez', 'Eric Lopez', 'Gerard Marti']
__license__ = "GPL"
__version__ = "1.0"

#################################
# EXPERIMENT CONFIGURATION
#################################

# Select between experiments. exp_methodology = 2 means color enhacement + ccl, exp_methodology = 1 means per image problem, exp_methodology = 0 means sliding window
exp_methodology = 2

#################################
# DESCRIPTOR
#################################

# Choose the descriptor, it can be LBP, HOG, GRAYPATCH or HAAR
featuresToExtract = ['HOG'] 

# Assign the number of features for each descriptor
if 'HOG' in featuresToExtract:
	num_features = 324
elif 'LBP' in featuresToExtract:
	num_features = 531	
elif 'HAAR' in featuresToExtract:
	num_features = 647		
else:
	num_features = 1024	
	
# DESCRIPTOR CONFIGURATION	

# LBP Parameters
lbp_win_shape = (16, 16)
lbp_win_step = lbp_win_shape[0]/2
lbp_radius = 1
lbp_n_points = 8 * lbp_radius
lbp_METHOD = 'nri_uniform' # "nri" means non-rotation invariant
lbp_n_bins = 59 # NRI uniform LBP has 59 values

# HOG Parameters
hog_orientations = 9
hog_pixels_per_cell = (8, 8)
hog_cells_per_block = (2, 2)
hog_normalise = True

#################################
# DATASET Settings
#################################

# Each folder is a class
if (exp_methodology == 2):
	data = ['F45', 'F49', 'F87', 'C11', 'A14', 'D1b_rechts_onder', 'zone_P3,5tmax_herhaling', 'Background', 'C3', 'E7', 'E1', 'E3', 'C31LEFT', 'C23', 'F50', 'C43', 'F19', 'A1B', 'A7B', 'A23', 'E9a_miva', 'uitgezonderd_plaatselijk_verkeer', 'D9', 'D7', 'rode pijl op wit vlak']
	trainingFolderPath = 'Datasets/DatasetSystem/Train' #TODO: Se tiene que modificar, estamos entrenando con el data set de test!
	num_train_images = 317
elif (exp_methodology == 1):
	data = ['Caution','DeadEnd','Giveway','HumpBlue','HumpRed','Intersection','NoEntry','NoParking','NoStopping','OneWay','Pedestrians','Roundabout','School','Stop']
	trainingFolderPath = 'Datasets/Perimage/Train'
	num_train_images = 784
else:
	data = ['Background','Circulares','Cuadradas','Triangulares']
	trainingFolderPath = 'Datasets/CropProject'
	num_train_images = 3580
num_classes = len(data)

# Configure the train dataset
FeaturesInPath = data

# Configure the test dataset
testFolderPath = 'Datasets/DatasetSystem/Test'
annotationsFolderPath = 'Datasets/DatasetSystem/Test/gt'

# Configure the output path from matlab (color enh + ccl)
test_omatlab_folder_path = 'Datasets/Crops'

# Configure the results
resultsFolder = 'Results/'

#################################
# MODEL settings
#################################

# XGBOOST implements an extreme Gradient Boosting
model = 'XGBOOST'

# Location of the model
modelFeatures = '-'.join(featuresToExtract)
modelPath = 'BModel/'+model+'_'+modelFeatures+'.model'

# Location of the process. data set
pDatasetPath = 'PData/Train_dataset_'+modelFeatures+'.data'
tDatasetPath = 'PData/Test_dataset_'+modelFeatures+'.data'

# xgBoost parameters
xgParam = {}
xgParam['nthread'] = 4
xgParam['silent'] = 1 # 0 means printing running messages, 1 means silent mode.

xgParam['objective'] = 'multi:softmax' # "multi:softmax" set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
xgParam['num_class'] = num_classes
xgParam['eval_metric'] = 'merror' # "merror": Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).

xgParam['eta'] = 0.02 # Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features. and eta actually shrinks the feature weights to make the boosting process more conservative.
xg_num_round = 200 # The number of rounds for boosting

xgParam['booster'] = 'gblinear' # Which booster to use

'''
xgParam['booster'] = 'gbtree'
xgParam['gamma'] = 0.36 # Minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, the more conservative the algorithm will be.
xgParam['min_child_weight'] = 1 # Minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.
xgParam['max_depth'] = int(num_features/4) # Maximum depth of a tree
xgParam['subsample'] = 0.5 # Subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting.
'''

#################################
# COLOR ENHACEMENT + CCL METHOD SETTINGS
#################################

negative_Class = 'Background'
ce_ccl_test_dataset = 'PData/Ce_ccl_test_dataset_'+modelFeatures+'.data'

#################################
# PER IMAGE METHOD SETTINGS
#################################

test_perImage_path = 'Datasets/Perimage/Test'
test_perImage_folders = data
test_perImage_dataset = 'PData/Test_perImage_dataset_'+modelFeatures+'.data'

#################################
# SLIDING WINDOW METHOD SETTINGS
#################################

# Assign the identifier of a signal in order to be detected
Cuadradas = ['B21', 'E9', 'F']
Triangulares = ['A', 'B1', 'B17', 'B3']

# Size of windows for the sliding window on the test images, multiple of 8!
window_shape = (32,32)
window_margin = 0
window_step = 8

# Downscale factor for the pyramid
downScaleFactor = 1.2

# Padding added to the test images. Used to detect pedestrians at the border of the image.
padding = 0

# Non-Maximum suppression overlap threshold
#   Two bounding boxes are considered the same
#   if their overlapping percentage exceeds this value.
nmsOverlapThresh = 0.5

# Values used in evaluation
decision_threshold_min = 0.7
decision_threshold_max = 1
decision_threshold_step = 0.02

# Percentage that the detections and the annotations have to overlap, to consider a detection correct.
annotation_min_overlap = 0.5