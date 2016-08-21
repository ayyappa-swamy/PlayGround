"""
	This module contains the methods used for processing the frames captured.
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
from scipy import stats
import statsHandler

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

# It calculates the coordinates of the hand portion in current frame
def getHandCoordinates(currentFrame, parameters):
	x = 0
	y = 0

	currentFrame = equalizeLuminance(currentFrame)
	currentFrame_hsv = cv.cvtColor(currentFrame, cv.COLOR_BGR2HSV)

	skinBlob = statsHandler.detectSkinPixels(currentFrame_hsv, parameters)
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
	print handImage_hsv.shape
	hue, sat, value = cv.split(handImage_hsv)

	nonZeroPixelIndices = np.nonzero(value)

	hue = hue /180.0
	sat  = sat/255.0

	s_mean, s_std = statsHandler.computeStats(sat,nonZeroPixelIndices)
	h_mean, h_std = statsHandler.computeStats(hue,nonZeroPixelIndices)

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
