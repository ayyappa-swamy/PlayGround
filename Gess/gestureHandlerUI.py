"""
	This module handles the UI functions. Its member functions capture the user hand for initialization.
	It also includes the function for capturing current frame
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
from scipy import stats
import statsHandler
import gestureHandler

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

#It is used to plot the coordinates in a separate window
def plotHandCentroid(x,y) :
	return

# It captures each frame from camera and plots the centroid of hand portion in input frame
def detectHand():
	handImage = getHandImage()
	cv.imwrite('temp.jpg',handImage)
	parameters = gestureHandler.getHandParameters(handImage)
	print parameters
	while True:
		ret, currentFrame = videoObj.read()
		[x, y] = gestureHandler.getHandCoordinates(currentFrame, parameters)

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
