import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
from scipy import stats

videoObj = cv.VideoCapture(0)

# It prompts the user to place the hand in the boundary shown
# The hand boundary is drawn by using a sample hand5.jpg image in workspace
# When the user places their hand, it is masked with a hand mask to retrieve the hand portion from current frame
def getHandImage():

	handImage = []
	hand = cv.imread('/home/ayyappa/Documents/PythonWorkspace/PlayGround/hand5.jpg')
	#cap = cv.VideoCapture(0)
	grayHand = cv.cvtColor(hand,cv.COLOR_BGR2GRAY)
	ret,maskImage = cv.threshold(grayHand,25,255,cv.THRESH_BINARY)

	maskHeight = maskImage.shape[0]
	maskWidth = maskImage.shape[1]

	maskGradient = cv.morphologyEx(maskImage, cv.MORPH_GRADIENT,cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)))

	while True:
		ret, frame = videoObj.read()

		frameWidth = frame.shape[1]
		frameHeight = frame.shape[0]

		blackImage = np.zeros((frameHeight,frameWidth),dtype=np.uint8)
		maskImageEx = np.zeros((frameHeight,frameWidth),dtype=np.uint8)

		#print "blackImage size is ", blackImage.shape
		#print "maskImageEx size is ", maskImageEx.shape

		blackImage[frameHeight/2-maskHeight/2:frameHeight/2+maskHeight/2,frameWidth/2-maskWidth/2:frameWidth/2+maskWidth/2] = maskGradient[:,:]
		maskImageEx[frameHeight/2-maskHeight/2:frameHeight/2+maskHeight/2,frameWidth/2-maskWidth/2:frameWidth/2+maskWidth/2] = maskImage[:,:]

		#cv.imshow('maskImage',maskImageEx)
		#cv.imshow('blackImage', blackImage)

		cv.imshow('added', cv.add(frame,cv.cvtColor(blackImage,cv.COLOR_GRAY2BGR)))

		handImage = cv.bitwise_and(frame,frame,mask = maskImageEx)
		#cv.imshow('and Image', handImage)

		if cv.waitKey(1) & 0xFF == ord('q') :
			break

	#cap.release()

	cv.imshow('finalhand',handImage)
	cv.waitKey(0)
	cv.destroyAllWindows()

	return handImage

# It computes the mean and standard deviation of input HSV image
# plane - parameter to select the plane => 0,1,2
def computeStats(handImage_hsv,dimension) :
	mean = 0
	std = 0
	mean = np.mean(handImage_hsv[:,:,dimension])
	std = np.std(handImage_hsv[:,:,dimension])
	return mean, std

# It equalizes the luminance of an image.
def equalizeLuminance(image):
	#image = cv.imread('/home/ayyappa/Documents/PythonWorkspace/PlayGround/temp.png')
	channels = []
	image_YCrCb = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)

	y,cr,cb = cv.split(image_YCrCb)
	y = cv.equalizeHist(y)

	image_YCrCb = cv.merge((y, cr, cb))
	#cv.imshow('equalized', image_YCrCb)
	#cv.waitKey(0)
	image = cv.cvtColor(image_YCrCb, cv.COLOR_YCR_CB2BGR)

	return image

# It detects the skin coloured pixels in the input frame by
# by calculating the gaussian probability
def detectSkinPixels(image, parameters):
	s_probability = stats.norm.pdf(image[:,:,1], parameters['s']['mean'], parameters['s']['std'])
	h_probability = stats.norm.pdf(image[:,:,0], parameters['h']['mean'], parameters['h']['std'])
	print "S Probability - max:", np.max(s_probability), " min: ",np.min(s_probability) # range 0.035 max inverting
	print "H Probability - max:", np.max(h_probability), " min: ",np.min(h_probability) # range 0.039 max working better
	#_#_#_#_#_#_#_#_#_#_ Book mark. equalization not much satisfying
	cv.imshow('s_probability',s_probability)
	cv.imshow('h_probability',h_probability)

# It calculates the coordinates of the hand portion in current frame
def getHandCoordinates(currentFrame, parameters):
	x = 0
	y = 0

	currentFrame = equalizeLuminance(currentFrame)
	currentFrame_hsv = cv.cvtColor(currentFrame, cv.COLOR_BGR2HSV)

	skinBlob = detectSkinPixels(currentFrame_hsv, parameters)
	#handBlob = detectHandPixels(skinBlob)

	""""# --> Helper for calculating probability from scipy import stats
	>>> # PDF of Gaussian of mean = 0.0 and std. deviation = 1.0 at 0.
	>>> stats.norm.pdf(0, loc=0.0, scale=1.0)
	0.3989422804014327 """

	#calculate centroid of hand blob

	return [x, y]

# It returns a parameter dictionary of mean and variances of each plane H,S
# by invoking respective functions for Histogram equalization -> calculate parameters
def getHandParameters(handImage):

	handImage = equalizeLuminance(handImage)

	handImage_hsv = cv.cvtColor(handImage, cv.COLOR_BGR2HSV)
	s_mean, s_std = computeStats(handImage_hsv, 1)
	h_mean, h_std = computeStats(handImage_hsv, 0)

	parameterDictionary = {
		"s" : {
			"mean" : s_mean,
			"std" : s_std
		},
		"h" : {
			"mean" : h_mean,
			"std" : h_std
		}
	}

	return parameterDictionary

#It is used to plot the coordinates in a separate window
def plotHandCentroid(x,y) :
	return

# It captures each frame from camera and plots the centroid of hand portion in input frame
def detectHand():
	handImage = getHandImage()
	cv.imwrite('temp.jpg',handImage)
	parameters = getHandParameters(handImage)
	print parameters
	while True:
		ret, currentFrame = videoObj.read()
		[x, y] = getHandCoordinates(currentFrame, parameters)

		#plotHandCentroid(x, y)
		if cv.waitKey(1) & 0xFF == ord('q') :
			break
	return 0

detectHand()
	#mask = getHandMask

	#handOutline = cv2.morphologyEx(pass the arguments)

	#displayFrame = currentFrame bitwise and with handOutline

	#take the input from user

	#ret, capturedFrame = videoObj.read()

	#convert the capturedFrame to YCbCr

	#Equalize Y component in capturedFrame

	#Find mean and variance of H, S components in equalised frame

	#Based on this calculate the gaussian probability and detect skin pixels

	#Better to use meanshift/convex hull/ optical flow/ some method for detecting hand shape

	#Find the centroid, plot the values using matplot lib
