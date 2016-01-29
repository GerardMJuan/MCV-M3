from PIL import Image, ImageDraw
import os

fileList = os.listdir(os.getcwd())
imagesList = filter(lambda element: '.jpg' in element, fileList)
for filename in imagesList:
	print 'Processing '+filename
	# load the image, create the mirrored image, and the result placeholder
	img    = Image.open(filename)
	mirror = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
	sz     = max(img.size + mirror.size)
	result = Image.new(img.mode, (sz,sz))
	result.paste(img, (0,0)+img.size)

	# now paste the mirrored image, but with a triangular binary mask
	mask = Image.new('1', mirror.size)
	draw = ImageDraw.Draw(mask)
	draw.polygon([0,0,0,sz,sz,sz], outline='white', fill='white')
	result.paste(mirror, (0,0)+mirror.size, mask)

	# clean up and save the result
	del mirror, mask, draw
	output = 'mirr_'+filename
	result.save(output)