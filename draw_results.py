#!/usr/bin/python
#Copyright 2015 CVC-UAB

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

__author__ = "Miquel Ferrarons, David Vazquez"
__copyright__ = "Copyright 2015, CVC-UAB"
__credits__ = ["Miquel Ferrarons", "David Vazquez"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Miquel Ferrarons"

import pickle
import os.path
from skimage import io
from Tools import drawing
import Config as cfg
from PIL import Image
import numpy as np
from Tools import nms

def run():
	if cfg.exp_methodology == 2:
		#Check which one is the negative class (background)
		negclass = cfg.data.index(cfg.negative_Class)
		
		#Open results file, it contains the 'candidates' structure.
		file = open(cfg.resultsFolder+'/Ce_ccl.results', 'r')
		candidates = pickle.load(file)
		if candidates['bboxes'] is None:
			print 'No signals found'
		else:
			#for each image in cfg.testFolderPath
			extension = '.jpg'
			#For each stored bounding box
			imagepath = '';
			img = None;
			for index in range(0, len(candidates['bboxes'])):
				i_image = candidates['bboxes'][index][0]
				i_box = candidates['bboxes'][index][-4:]
				i_prediction = candidates['prediction'][index]
				
				if int(i_prediction) != int(negclass): # Do not print boxes predicted as background
					predicted_sign = cfg.data[int(i_prediction)]
					real_sign = cfg.data[int(candidates['bboxes'][index][1])]
					
					results_curr_img_path = 'Results/' + i_image + extension
					
					if os.path.isfile(results_curr_img_path): # If exists
						#print 'exist!'
						imagepath = results_curr_img_path;
					else:
						#print 'NOT exist!'
						imagepath = cfg.testFolderPath + '/' + i_image + extension;

					img = Image.open(imagepath)
						
					if img is not None:
						print 'Drawing results on ' + i_image + extension
						img = drawing.drawResultsOnImage(img, i_box, predicted_sign, real_sign)
						io.imsave(results_curr_img_path, img)
		file.close()
if __name__ == '__main__':
    run()