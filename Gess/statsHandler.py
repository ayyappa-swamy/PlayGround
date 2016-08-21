"""
	This module contains the functions for performing statistical operations
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
from scipy import stats

# It computes the mean and standard deviation of input HSV image
# plane - parameter to select the plane => 0,1,2
def computeStats(array,indices) :
	mean = 0
	std = 0

	mean = np.mean(array[indices])
	std = np.std(array[indices])

	return mean, std

# It detects the skin coloured pixels in the input frame by
# by calculating the gaussian probability
def detectSkinPixels(image, parameters):
	hue , sat , val = cv.split(image)
	hue = hue/180.0
	sat = sat/255.0
	s_probability = stats.norm.pdf(sat, parameters['s']['mean'], parameters['s']['std'])
	h_probability = stats.norm.pdf(hue, parameters['h']['mean'], parameters['h']['std'])

	#print "S Probability - max:", np.max(s_probability), " min: ",np.min(s_probability) # range 0.035 max inverting
	#print "H Probability - max:", np.max(h_probability), " min: ",np.min(h_probability) # range 0.039 max working better
	#_#_#_#_#_#_#_#_#_#_ Book mark. equalization not much satisfying
	cv.imshow('s_probability',s_probability)
	cv.imshow('h_probability',h_probability)

