import cv2 as cv
import numpy as np

capVideoObj = cv.VideoCapture(0)

while(1):
	ret, frame = capVideoObj.read()
	hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

	skin_up = np.uint8([40,255,255])
	skin_lower = np.uint8([20,210,5])

	mask = cv.inRange(hsvFrame,skin_lower,skin_up)

	#result = cv.bitwise_and(frame,frame,mask=mask)

	cv.imshow('hsvImage',hsvFrame)

	cv.imshow('Mask',mask)

	#cv.imshow('skin',result)

	if cv.waitKey(10) == ord('q') :
		break

capVideoObj.release()
cv.destroyAllWindows()


#cv.imwrite('newImage2.png',frame)


##My skin hsv values are
# skin_up = np.array([40,255,90])
# skin_lower = np.array([20,210,5])

#ret, frame = capVideoObj.read()
#hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

#mask = cv.inRange(hsvFrame,skin_lower,skin_up)

#result = cv.bitwise_and(frame,frame,mask= mask)

#cv.imshow('hsvImage',hsvFrame)

#cv.imshow('Mask',mask)

#cv.imshow('skin',result)

#k = cv.waitKey()

#cv.destroyAllWindows()
