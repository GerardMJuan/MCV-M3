# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 14:17:25 2016

@author: Eric López
"""


from skimage import io,transform
import os
import skimage.util as util

inputFolder = 'DatasetSystem'
gtFolder = 'DatasetSystem/gt'
resultFolder = 'Circulares'

#To avoid border problems
p = 50
padding = ((p,p),(p,p),(0,0))

#Creates a folder for the results
if not os.path.exists(resultFolder):
    os.makedirs(resultFolder)

lengthResize = 32
margin = 4

#List all files
fileList = os.listdir(inputFolder)
#Select only files that end with .jpg
imagesList = filter(lambda element: '.jpg' in element, fileList)

for filename in imagesList:
    imagePath = inputFolder + '/' + filename
    
    image = io.imread(imagePath)
    image = util.pad(image,padding,'reflect')
    
    name = os.path.splitext(filename)[0]
    
    gtPath = gtFolder+'/gt.'+name+'.txt'
    
    with open(gtPath,'r') as fp:
        numSignal = 0;
        for line in fp:
            #Annotates the two corners of the bounding box
            y1, x1, y2, x2, string = line.split(' ',4)
            
            
            if (string[0]=='C')| (string[0]=='D') | ((string[0]== 'E')& (string[1]!='9')):#Circulares
            #if (string[0]=='F')| (string[0:1]=='E9') | (string[0:2]== 'B21'): #Cuadradas
            #if (string[0]=='A')| ((string[0]== 'B')& ((string[1:2]!='21')| (string[1]!='9'))): #Triangulares
                w = abs(float(y1)-float(y2))
                h = abs(float(x1)-float(x2))
                d = round(max(w,h))
                yc = float(y1)+w/2+p
                xc = float(x1)+h/2+p
                toAdd = d/lengthResize*margin+d/2 #Resize ratio X margin + distance to the center
                
                #Counter to correctly save the signal
                numSignal = numSignal+1        
                
                crop = image[round(yc-toAdd):round(yc+toAdd),round(xc-toAdd):round(xc+toAdd)]
                crop = transform.resize(crop,(lengthResize,lengthResize))
                io.imsave(resultFolder+'/'+name+'_'+str(numSignal)+'.jpg', crop)
                del crop
