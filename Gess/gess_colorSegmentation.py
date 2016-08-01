import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
import scipy as sp

videoObj = cv.videoCapture(0)

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

def computeMeanVariance(handImage) :
	mean = 0
	variance = 0
	return mean, variance

def equalizeLuminance(image_YCrCb):
	return image_YCrCb

def getHandCoordinates(currentFrame, parameters):


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
