import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
import scipy as sp

videoObj = cv.videoCapture(0)

# It prompts the user to place the hand in the boundary shown
# The hand boundary is drawn by using a sample hand5.jpg image in workspace
# When the user places their hand, it is masked with a hand mask to retrieve the hand portion from current frame
def getHandImage:

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

	cap.release()

	cv.imshow('finalhand',handImage)
	cv.waitKey(0)
	cv.destroyAllWindows()

	return handImage

# It computes the mean and variance of input HSV image
# plane - parameter to select the plane => 's', 'h', 'v'
def computeMeanVariance(handImage_hsv,plane) :
	mean = 0
	variance = 0
	mean = numpy.mean(array)
	variance = numpy.var(array) #Think if we can directly use standard deviation numpy.std
	return mean, variance

# It equalizes the luminance of the YCrCb image.
def equalizeLuminance(image_YCrCb):
	image_YCrCb[:,:,0] = cv.equalizeHist(image_YCrCb[:,:,0])
	return image_YCrCb

# It calculates the coordinates of the hand portion in current frame
def getHandCoordinates(currentFrame, parameters):
	x = 0
	y = 0

	skinBlob = detectSkinPixels(currentFrame, parameters)
	handBlob = detectHandPixels(skinBlob)

	""""# --> Helper for calculating probability from scipy import stats
	>>> # PDF of Gaussian of mean = 0.0 and std. deviation = 1.0 at 0.
	>>> stats.norm.pdf(0, loc=0.0, scale=1.0)
	0.3989422804014327 """

	#calculate centroid of hand blob

	return [x, y]

# It returns a parameter dictionary of mean and variances of each plane H,S
# by invoking respective functions for Histogram equalization -> calculate parameters
def getHandParameters(handImage):

	handImage_ycrcb = cv.cvtColor(handImage, cv.COLOR_BGR2YCrCb)

	handImage_ycrcb = equalizeLuminance(handImage_ycrcb)

	handImage = cv.cvtColor(handImage_ycrcb, cv.COLOR_YCrCb2BGR)

	handImage_hsv = cv.cvtColor(handImage, cv.COLOR_BGR2HSV)

	s_mean, s_variance = computeMeanVariance(handImage_hsv, 's')
	h_mean, h_variance = computeMeanVariance(handImage_hsv, 'h')

	parameterDictionary = {
		"s" : {
			"mean" : s_mean,
			"variance" : s_variance
		},
		"h" : {
			"mean" : h_mean,
			"variance" : h_variance
		}
	}

	return parameterDictionary

#It is used to plot the coordinates in a separate window
def plotHandCentroid(x,y) :
	return

# It captures each frame from camera and plots the centroid of hand portion in input frame
def detectHand():

	handImage = getHandImage()

	parameters = getHandParameters(handImage)

	while True:
		ret, currentFrame = videoObj.read()

		[x, y] = getHandCoordinates(currentFrame, parameters)

		plotHandCentroid(x, y)

		if cv.waitKey(1) & 0xFF == ord('q') :
			break

	return 0

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
