from PIL import Image
import os, sys

path = "C:\Users\ssancho\Documents\sergiuni\UAB\M3\Traffic_Sign\Datasets\DatasetSystem\\"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        #print 'item >> '+item
        if os.path.isfile(path+item):
			im = Image.open(path+item)
			im.save('01.002785.png')

resize()