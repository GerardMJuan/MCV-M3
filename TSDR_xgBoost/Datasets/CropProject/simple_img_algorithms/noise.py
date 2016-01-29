from PIL import Image, ImageDraw
import os
import numpy as np
import os

fileList = os.listdir(os.getcwd())
imagesList = filter(lambda element: '.jpg' in element, fileList)
for filename in imagesList:
	print 'Processing '+filename
	# load the image, create the mirrored image, and the result placeholder
	img    	= Image.open(filename)
	
	row 	= 32
	col 	= 32
	ch 		= 1
	mean 	= 0
	var 	= 0.1
	sigma 	= var**0.5
	gauss 	= np.random.normal(mean,sigma,(row,col,ch))
	gauss 	= gauss.reshape(row,col,ch)
	noisy 	= img + gauss
	
	#result 	= Image.new(img.mode, (row,col))
	#result.paste(img, (0,0)+img.size)
	
	# clean up and save the result
	#del mirror, mask, draw
	#output = 'noisy_'+filename
	#result.save(output)
	im = Image.fromarray(noisy, mode='RGB')
	im.save('yo.jpg')

'''	
def noisy(noise_typ,image):
	if noise_typ == "gauss":
		row,col,ch= image.shape
		mean = 0
		var = 0.1
		sigma = var**0.5
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		gauss = gauss.reshape(row,col,ch)
		noisy = image + gauss
		return noisy
	elif noise_typ == "s&p":
		row,col,ch = image.shape
		s_vs_p = 0.5
		amount = 0.004
		out = image
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt))
		for i in image.shape]
		out[coords] = 1

		# Pepper mode
		num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper))
		for i in image.shape]
		out[coords] = 0
		return out
	elif noise_typ == "poisson":
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy
	elif noise_typ =="speckle":
		row,col,ch = image.shape
		gauss = np.random.randn(row,col,ch)
		gauss = gauss.reshape(row,col,ch)        
		noisy = image + image * gauss
		return noisy
'''