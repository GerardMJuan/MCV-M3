from PIL import Image, ImageDraw
import os

fileList = os.listdir(os.getcwd())
imagesList = filter(lambda element: '.jpg' in element, fileList)
for filename in imagesList:

	##### FLIP HORIZONTAL #####
	print 'Processing '+filename
	# load the image, create the mirrored image, and the result placeholder
	img    	= Image.open(filename)
	mirror 	= img.transpose(Image.FLIP_LEFT_RIGHT)
	img 	= mirror
	sz     	= max(img.size + mirror.size)
	result 	= Image.new(img.mode, (sz,sz))
	result.paste(img, (0,0)+img.size)

	# clean up and save the result
	del mirror#, img
	output 	= 'flipH_'+filename
	result.save(output)
	##### END FLIP HORIZONTAL #####