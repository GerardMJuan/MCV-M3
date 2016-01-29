	from PIL import Image
import os, sys

path = "C:\Users\Adriana\Desktop\M3\Block1\Traffic_Sign\Datasets\Testing\Test"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((30,30), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=90)

resize()