# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 22:28:44 2016

@author: Eric López

"""

from skimage import io,transform
import os
import numpy.random as rd

inputFolder = 'NonTSImages/TestingBG'
resultFolder = 'Background'


#Creates a folder for the results
if not os.path.exists(resultFolder):
    os.makedirs(resultFolder)

lengthResize = 32

#List all files
filesList = os.listdir(inputFolder)
imagesList = filter(lambda element: '.jp2' in element, filesList)

for filename in imagesList:
    imagePath = inputFolder + '/' + filename
    
    image = io.imread(imagePath)
    size = image.shape
    
    #Calculate the maximum window in order to not get out from the image
    maxWindow = min(size[0],size[1])/2
    
    name = os.path.splitext(filename)[0]
    i=0
    #We have chosen 4 sample background per image, but we can change it
    while i<4:
        #Top-left corner of the window
        y1 = rd.uniform(0, size[0])
        x1 = rd.uniform(0, size[1])
        
        #Lenght of side of the square window
        l = rd.uniform(32, maxWindow)
        
        #If we exceed the image limits in one direction we go to the contrary direction
        if y1+l>size[0]:
            y1 = y1-l
            y2 = y1+l
        else:
            y2 = y1+l
            
        if x1+l>size[1]:
            x1 = x1-l
            x2 = x1+l
        else:
            x2 = x1+l
        
        crop = image[y1:y2,x1:x2]
        crop = transform.resize(crop,(lengthResize,lengthResize))
        
        i = i+1
        
        io.imsave(resultFolder+'/'+name+'_'+str(i)+'.jpg', crop)
        del crop
            


