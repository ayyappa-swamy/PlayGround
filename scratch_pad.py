import cv2 as cv
import numpy as np

## reading and displaying an image file
#inputImage = cv.imread('/home/ayyappa/Pictures/swamy.jpg',0)
#cv.imshow('image',inputImage)
#cv.waitKey(0)


## capturing a video and displaying in GRAYSCALE
#cap = cv.VideoCapture(0)

#while(True):

	#ret, frame = cap.read()

	#gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	#cv.imshow('frame', gray)

	#if cv.waitKey(1) & 0xFF == ord('q') :
		#break

#cap.release()
#cv.destroyAllWindows()

# Capturing and Saving a horizontally flipped video
#cap = cv.VideoCapture(0)
#fourcc = cv.cv.CV_FOURCC(*'XVID')
#out = cv.VideoWriter('output.avi',fourcc,20.0,(640,480))

#while(cap.isOpened()):

	#ret, frame = cap.read()

	#if ret == True:
		##frame = cv.flip(frame,1) # 1 => flip along Y axis,   0 => X axis, -1 => both axes

		#gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

		#out.write(gray)

		#cv.imshow('frame',gray)

		#if cv.waitKey(1) & 0xFF == ord('q') :
			#break
	#else:
		#break

#cap.release()
#cv.destroyAllWindows()

##Simple mouse callback demo
#def draw_circle(event,x,y,flags,param):
	#if event == cv.EVENT_LBUTTONDBLCLK :
		#cv.circle(img, (x,y),100,(255,0,0),-1)

#img = np.zeros((512,512,3),np.uint8)
#cv.namedWindow('image')
#cv.setMouseCallback('image',draw_circle)

#while(1):
	#cv.imshow('image',img)
	#if cv.waitKey(20) & 0xFF == 27:
		#break
#cv.destroyAllWindows()

##Advanced demo
#drawing = False
#mode = True
#ix, iy = -1, -1

#def draw_circle(event,x,y,flags,param) :
	#global ix,iy,drawing,mode

	#if event == cv.EVENT_LBUTTONDOWN:
		#drawing = True
		#ix,iy = x,y
	#elif event == cv.EVENT_MOUSEMOVE:
		#if drawing == True:
			#if mode == True:
				#cv.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
			#else :
				#cv.circle(img,(x,y),5,(0,0,255),-1)
	#elif event == cv.EVENT_LBUTTONUP:
		#drawing = False
		#if mode == True:
			#cv.rectangle(img,(ix,iy),(x,y),(0,255,0),3)
		#else:
			#cv.circle(img,(x,y),5,(0,0,255),-1)

#img = np.ones((512,512,3),np.uint8)
#cv.namedWindow('image')
#cv.setMouseCallback('image',draw_circle)

#while(1):
	#cv.imshow('image',img)
	#k = cv.waitKey(1) & 0xFF
	#if k == ord('m'):
		#mode = not mode
	#elif k == 27:
		#break

#cv.destroyAllWindows()

#bitwise Operations


#img1 = cv.imread('/home/ayyappa/Documents/PythonWorkspace/PlayGround/test.jpg')
#img2 = cv.imread('/home/ayyappa/Documents/PythonWorkspace/PlayGround/opencv_logo.png')
#rows,cols,channels = img2.shape
#roi = img1[0:rows, 0:cols]

#print img1.shape
#print "size of img 2 is ", img2.shape

#img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
#ret,mask = cv.threshold(img2gray,5,255,cv.THRESH_BINARY)
#mask_inv = cv.bitwise_not(mask)

#cv.imshow('mask',mask)
#cv.imshow('mask_inv', mask_inv)
#cv.imshow('gray logo', img2gray)

#img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
#cv.imshow('background of img1',img1_bg)

#img2_fg = cv.bitwise_and(img2,img2,mask = mask)
#cv.imshow('foreground of img2', img2_fg)

#dst = cv.add(img1_bg, img2_fg)
#img1[0:rows, 0:cols] = dst

#cv.imshow('result', img1)
#cv.waitKey(0)
#cv.destroyAllWindows()

# capturing a video and displaying in GRAYSCALE
#cap = cv.VideoCapture(0)

#flag = True
#prevFrame = []
#diffImage = []

#while(True):

	#ret, frame = cap.read()

	##gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	#frameWidth = frame.shape[1]
	#frameHeight = frame.shape[0]

	#if flag :
		#prevFrame = frame * 0
		#flag = False

	##cv.rectangle(frame - prevFrame, (frameWidth/2-30,frameHeight/2-30), (frameWidth/2+30,frameHeight/2+30), (0,255,0), 2)
	#diffImage = cv.cvtColor(cv.subtract(frame, prevFrame), cv.COLOR_BGR2GRAY)

	#cv.imshow('frame', frame)

	#prevFrame = frame

	#if cv.waitKey(1) & 0xFF == ord('q') :
		#break

#cv.imwrite('handImage.png',frame)
#--------------------------------------------------------------------------------
##the below code captures the hand image and gives the cropped hand by masking
#--------------------------------------------------------------------------------
#hand = cv.imread('/home/ayyappa/Documents/PythonWorkspace/PlayGround/hand5.jpg')
#cap = cv.VideoCapture(0)
#grayHand = cv.cvtColor(hand,cv.COLOR_BGR2GRAY)
#ret,maskImage = cv.threshold(grayHand,25,255,cv.THRESH_BINARY)

#maskHeight = maskImage.shape[0]
#maskWidth = maskImage.shape[1]

#maskGradient = cv.morphologyEx(maskImage, cv.MORPH_GRADIENT,cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)))

#while True:
	#ret, frame = cap.read()

	#frameWidth = frame.shape[1]
	#frameHeight = frame.shape[0]

	#blackImage = np.zeros((frameHeight,frameWidth),dtype=np.uint8)
	#maskImageEx = np.zeros((frameHeight,frameWidth),dtype=np.uint8)

	##print "blackImage size is ", blackImage.shape
	##print "maskImageEx size is ", maskImageEx.shape

	#blackImage[frameHeight/2-maskHeight/2:frameHeight/2+maskHeight/2,frameWidth/2-maskWidth/2:frameWidth/2+maskWidth/2] = maskGradient[:,:]
	#maskImageEx[frameHeight/2-maskHeight/2:frameHeight/2+maskHeight/2,frameWidth/2-maskWidth/2:frameWidth/2+maskWidth/2] = maskImage[:,:]

	##cv.imshow('maskImage',maskImageEx)
	##cv.imshow('blackImage', blackImage)

	#cv.imshow('added', cv.add(frame,cv.cvtColor(blackImage,cv.COLOR_GRAY2BGR)))

	#img_fg = cv.bitwise_and(frame,frame,mask = maskImageEx)
	##cv.imshow('and Image', img_fg)

	#if cv.waitKey(1) & 0xFF == ord('q') :
		#break

#cap.release()

#cv.imshow('finalhand',img_fg)
#cv.waitKey(0)
#cv.destroyAllWindows()
aa = 1
bb = 2

aaa = 33
bbb = 44


a = {
	"c" : {
		"aa" : aa,
		"bb" : bb
	},
	"d" : {
		"aaa" : aaa,
		"bbb" : bbb
	}
}

print a['c'],
print a['d'],
print a['c']['bb'],
print a['d']['aaa']
