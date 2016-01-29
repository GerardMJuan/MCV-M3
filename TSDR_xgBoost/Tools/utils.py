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

import numpy as np

# Reads the annotations in the GT format [Y1, X1, Y2, X2]
# And returns the bounging boxes as [TLx,TLy,BRx,BRy]
#   TLx X1 :Top-Left X,
#   TLy Y2 :Top-Left Y,
#   BRx X2 :Bottom-Right X,
#   BRy Y1 :Bottom-Right y,
def readINRIAAnnotations(annotationsPath):

    annotatedBoxes = None

    with open(annotationsPath,'r') as fp:
        for line in fp:

            #Inria annotates the center of the pedestrian, and the width and height
            #xc, yc, w, h, string = line.split(' ', 4 )
            y1c, x1c, y2c, x2c, string = line.split(' ', 4 )
            x1 = float(x1c)
            y1 = float(y1c)
            x2 = float(x2c)
            y2 = float(y2c)
            bbox = (x1, y1, x2, y2)
            #bbox = (x1, y2, x2, y1)
            
            if annotatedBoxes is not None:
                annotatedBoxes = np.vstack((bbox, annotatedBoxes))
            else:
                annotatedBoxes = np.array([bbox])
    return annotatedBoxes
