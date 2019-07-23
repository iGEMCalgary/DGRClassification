#Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import math
import queue
import random
from sklearn.preprocessing import MinMaxScaler


#determine the difference between the current time and the given start time
def timeEndTime(startTime):
	endTime = time.time()
	deltaTime = endTime - startTime
	if deltaTime % 60 < 1:
		timeString = "Time: {:5.3f} milliseconds.".format((deltaTime%60)*1000)
	else:
		timeString = "Time: {} minutes, {:5.3f} seconds.".format(math.floor(deltaTime/60.0), deltaTime % 60)
	
	return timeString

#Read and return the 3-channel BGR image at the path given.
def readImage(path):
	return cv2.imread(path, cv2.IMREAD_COLOR)
	
#Display the given image
def displayImg(img, cmap='bgr', title='', block=True):
	plt.title(title)
	
	if cmap=='bgr':
		b, g, r = cv2.split(img)
		img = cv2.merge([r, g, b])
		plt.imshow(img)
	else:
		plt.imshow(img, cmap=cmap)
	
	if block == True:
		plt.show()
	else:
		plt.show(block=block)
		plt.pause(0.2)
	plt.close()
	
#Returns an image that is the flat average color of the given image
def averageImgColor(img):
	average = img.mean(axis=0).mean(axis=0)
	averageImg = np.ones(shape=img.shape, dtype=np.uint8) * np.uint8(average)
	return averageImg

#Return an image displaying the differences in color between a given img and a colour
def differenceImgColor(imgP, colorChipP, dgr='rel'):
	img = convertBRGToHSV(imgP)
	colorChip = convertBRGToHSV(colorChipP)
	
	displayImg(img, cmap='hsv', title='HSV img')
		
	diffImg = np.ones(shape=img.shape[:2])
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			diffImg[i][j] = hueDistanceGreen(img[i][j], colorChip[0][0], mode=dgr)
		
	displayImg(diffImg, cmap='gray', title='Distance Pure Img')
	
	diffImgCopy = np.zeros_like(diffImg)
	diffImgCopy[0] = np.array([0 for i in range(diffImgCopy.shape[1])])
	diffImgCopy[1] = np.array([90 for i in range(diffImgCopy.shape[1])])
		
	diffScaler = MinMaxScaler()
	diffScaler.fit(diffImgCopy)	
	
	diffImg = diffScaler.transform(diffImg)
	
	#displayImg(diffImgCopy, cmap='gray', title='Modified Distance Img Copy')

	diffImg = diffImg * 255
	diffImg = diffImg.astype(np.uint8)
	diffImg = reverseImg(diffImg)
	
	return diffImg
	
#Calculate the distance between pixels in color	
def threeChannelDistance(pxl1, pxl2):
	#convert pixels to float channels
	pxl1 = [float(pxl1[0]), float(pxl1[1]), float(pxl1[2])]
	pxl2 = [float(pxl2[0]), float(pxl2[1]), float(pxl2[2])]

	return math.sqrt((pxl1[0] - pxl2[0])**2 + (pxl1[1] - pxl2[1])**2 + (pxl1[2] - pxl2[2])**2)


def hueDistanceGreen(pxl1, colorChip, mode='rel'):
	
	
	#convert pixels to float channels
	pxl1 = [float(pxl1[0]), float(pxl1[1]), float(pxl1[2])]
	pxl2 = [float(colorChip[0]), float(colorChip[1]), float(colorChip[2])]
	
	if pxl1[0]==0 and pxl1[1]==0 and pxl1[2]==0:
			return 90
	
	if mode=='rel':
	
		hueDiffWeight = 1.1

		#Distance penalties for gray = 1.4(x-70/2)
		grayPenalty = lambda x: 1.4*(x-70/2) if x <= 70/2 else 0
				
		if pxl1[0] > 70/2 and pxl1[0] <= 130/2:
			return grayPenalty(pxl1[1])
		elif pxl1[0] <= 70/2:
			return min([hueDiffWeight*abs(pxl1[0] - 70/2), hueDiffWeight*abs(180-(pxl1[0] - 70/2))]) + grayPenalty(pxl1[1])
		else:
			return min([hueDiffWeight*abs(pxl1[0] - 130/2), hueDiffWeight*abs(180-(pxl1[0] - 130/2))]) + grayPenalty(pxl1[1])
	else:
		hueDiffWeight = 1.8
		
		#Distance penalties for being brighter than pxl2 is straight to max 90
		#Otherwise penalties exist for being less saturated offset by being darker.
		#Penalty = 1.0(sat1 - sat2) + 1.0(bri1 - bri2)
		brighterPenalty = lambda x: max([1.0*(x[0][1]-x[1][1])+1.0*(x[0][2]-x[1][2]),0]) if x[0][2] <= x[1][2] else 90
		"""		
		hue2Width = 10
		
		if pxl1[0] > (pxl2[0]-hue2Width)/2 and pxl1[0] <= (pxl2[0]+hue2Width)/2:
			return brighterPenalty((pxl1, pxl2))
		elif pxl1[0] <= (pxl2[0]-hue2Width)/2:
			return min([min([abs(pxl1[0] - (pxl2[0]-hue2Width)/2), 180-(pxl1[0] - (pxl2[0]-hue2Width)/2)]) + brighterPenalty((pxl1, pxl2)), 90])
		else:
			return min([min([abs(pxl1[0] - (pxl2[0]+hue2Width)/2), 180-(pxl1[0] - (pxl2[0]-hue2Width)/2)]) + brighterPenalty((pxl1, pxl2)),90])
		"""
		
		if pxl1[0] > 70/2 and pxl1[0] <= 130/2:
			return brighterPenalty((pxl1, pxl2))
		elif pxl1[0] <= 70/2:
			return min([min([hueDiffWeight*abs(pxl1[0] - 70/2), hueDiffWeight*abs(180-(pxl1[0] - 70/2))]) + brighterPenalty((pxl1, pxl2)), 90])
		else:
			return min([min([hueDiffWeight*abs(pxl1[0] - 130/2), hueDiffWeight*abs(180-(pxl1[0] - 130/2))]) + brighterPenalty((pxl1, pxl2)), 90])
		
		
		
		
		
		
		


#Make all white black and black white
def reverseImg(diffImg):
	return -1 * diffImg + 255 

#Convert a BGR image into a hsv img
def convertBRGToHSV(img):
	#Converting from HSV in plt to real HSV:
	#H = plt.H * 2
	#S = plt.S / 2.55
	#V = plt.V / 2.55
	return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
#Create a mask using thresholds
def maskImg(img, msk, threshold, background=True):
	blurredImg = np.zeros_like(img)
	if background == True:
		blurredImg = cv2.blur(img.copy(), (5,5))
		for i in range(1):
			blurredImg = cv2.blur(blurredImg, (5,5))
		blurredImg = adjustGamma(blurredImg, gamma = 0.6)

	if len(msk.shape) == 2:
		maskedImg = blurredImg.copy()
		maskedImg[msk >= threshold] = img[msk >= threshold]
	else:
		maskedImg = np.where(msk >= threshold, img, blurredImg)

	return maskedImg

#Adjust the gamme value of an image
def adjustGamma(img, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(img, table)

#Mark background and foreground pixels in the image
def markBackForeGround(img):
	startTime = time.time()
	
	#Parameters
	cooldownCap = 5
	
	#Create a blank grayscale mask of zeros 
	markerMsk = np.zeros_like(img[:,:,0]).astype(np.int32)
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	# Dictate the background and set the markers to 1
	#i = y from top to bot, j = x from left to right
	cooldownCur = cooldownCap
	
	for i in range(markerMsk.shape[0]):
		for j in range(markerMsk.shape[1]):
			if cooldownCur == 0: 
				if isBackground(imgHSV[i][j]):
					#Background marker
					markerMsk[i][j] = 1
				else:
					# Dictate the area of interest		
					#Each tag that hits a smear gets a smear
					markerMsk[i][j] = 255
				cooldownCur = cooldownCap
			cooldownCur = cooldownCur - 1
	
	displayImg(markerMsk, cmap='jet', title='marker mask')
	displayImg(maskImg(img, markerMsk, 254), title = 'marker mask')
	
	print("Initial marking finished. " + timeEndTime(startTime))
	
	return markerMsk

#Create BW watershed mask 
def createWatershedMsk(img, markerMsk):
	startTime = time.time()
	
	markerMskCopy = markerMsk.copy()
	
	#Initial watershed mask estimate:
	#Make the background black, and what we want to keep white
	watershedMsk = cv2.watershed(img, markerMskCopy)

	watershedMsk[watershedMsk == 1] = 0
	watershedMsk[watershedMsk > 1] = 255
	
	displayImg(watershedMsk, cmap='gray', title='Initial Watershed Mask')
	
	watershedMsk = fillHoles(watershedMsk.copy())
	
	displayImg(watershedMsk, cmap='gray', title='Filled Watershed Mask')
	displayImg(maskImg(img, watershedMsk, 200), title='Filled Watershed Mask')
	
	print("Watershed mask creation finished. " + timeEndTime(startTime))
	
	return watershedMsk

#Determines if a given HSV pixel is background.
def isBackground(imgPxl):
	svConv = lambda x: 0.01*x*255
	hConv = lambda x: x/2.
	greenHue = lambda x: x > hConv(70) and x < hConv(130)
	
	#Convert imgPxl to int array
	imgPxl = [int(imgPxl[0]), int(imgPxl[1]), int(imgPxl[2])]	
	
	#background = GRAY (not saturated)
	notSaturated = imgPxl[1] < svConv(35)
	
	#exceptions exist for GREEN hue but almost saturated enough
	greenAlmostNotSaturated = greenHue(imgPxl[0]) and imgPxl[1] > svConv(30)
	
	#exceptions exist for GREEN hue and not saturated but very dark
	greenVeryDark = greenHue(imgPxl[0]) and imgPxl[2] < svConv(25)
	
	"""
	#background = GRAY hue (blueish)
	isBlueish = imgPxl[0] < hConv(50) or imgPxl[0] > hConv(150)
	
	#background = GREEN hue and very unsaturated
	isGreyGreen = imgPxl[0] > hConv(70) and imgPxl[0] < hConv(130) and imgPxl[1] < svConv(10)
	
	if notSaturated: return notSaturated
	return (notSaturated and isBlueish) or isGreyGreen
	"""
	return notSaturated and not greenAlmostNotSaturated and not greenVeryDark
	
	
#Fill in small holes (val <= 0)in the image.
#maxSize = 85
def fillHoles(img, maxSize = 524):
	#Use floodfill and fill in the image
	i = j = 1
	while i < img.shape[0]-1:
		while j < img.shape[1]-1:
			if fillCriteria(img, i, j):
				img = floodfill(img, i, j, maxSize)
			removeSinglePixel(img, i, j, 255)
			j = j+1
			
		i = i+1
		j = 1
	
	return img

#Return true if the current pixel should be filled in.
#returns false if the pxl's cross area is all blank.
def fillCriteria(img, i, j, threshold=1):
	return img[i][j]<=threshold and (img[i-1][j]>threshold or img[i+1][j]>threshold or img[i][j-1]>threshold or img[i][j+1]>threshold) 

def floodfill(img, i, j, maxSize, replacedC = 0, replacementC = 255):
	#Floodfill algorithm
	if(i < 0 or j < 0 or i == img.shape[0] or j == img.shape[1] or img[i][j] > replacedC):
		return img
	
	imgC = img.copy()
	
	imgC[i][j] = replacementC
	
	Q = queue.Queue()	
	Q.put((i,j))
	
	lengthCounter = 0
	
	while not Q.empty():
		lengthCounter = lengthCounter + 1
		if lengthCounter > maxSize:
			return img
		
		n = Q.get()
		#west
		if(n[1] > 0 and imgC[n[0]][n[1]-1] <= replacedC):
			imgC[n[0]][n[1]-1] = replacementC
			Q.put((n[0],n[1]-1))
		#east
		if(n[1] < imgC.shape[1] - 2 and imgC[n[0]][n[1]+1] <= replacedC):
			imgC[n[0]][n[1]+1] = replacementC
			Q.put((n[0],n[1]+1))
		#north
		if(n[0] > 0 and imgC[n[0]-1][n[1]] <= replacedC):
			imgC[n[0]-1][n[1]] = replacementC
			Q.put((n[0]-1,n[1]))
		#south
		if(n[0] < imgC.shape[0]-2 and imgC[n[0]+1][n[1]] <= replacedC):
			imgC[n[0]+1][n[1]] = replacementC
			Q.put((n[0]+1,n[1]))
			
	return imgC 

#Removes a pixel that is solitary from an image)
def removeSinglePixel(img, i, j, threshold):
	if img[i][j] >= threshold and img[i-1][j]<threshold and img[i+1][j]<threshold and img[i][j-1]<threshold and img[i][j+1]<threshold:
		img[i][j] = 0

#Display statistical data about a give seed
def displaySeedInfo(seedFul, seedRel, seedDGR):
	areaFul = areaRel = areaDGR = 0
	
	for i in range(seedFul.shape[0]):
		for j in range(seedFul.shape[1]):
			if seedFul[i][j][0]!=0 and seedFul[i][j][1]!=0 and seedFul[i][j][2]!=0:
				areaFul = areaFul + 1
			if seedRel[i][j][0]!=0 and seedRel[i][j][1]!=0 and seedRel[i][j][2]!=0:
				areaRel = areaRel + 1
			if seedDGR[i][j][0]!=0 and seedDGR[i][j][1]!=0 and seedDGR[i][j][2]!=0:
				areaDGR = areaDGR + 1
	
	if areaRel > 0:
		dgrFrac = float(areaDGR)/float(areaRel)	
		print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		print("The full area of the seed is: {} pixels.".format(areaFul))
		print("The relevant/non-hull area of the seed is: {} pixels.".format(areaRel))
		print("The DGR area of the seed is: {} pixels.".format(areaDGR))
		print("The DGR % of the seed is: {:5.3f}% DGR".format(dgrFrac*100))
		print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	else:
		print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		print("No seed smear detected.")
		print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		
#Return the grade of canola based on fraction of dgr seeds.
def gradeCanola(dgrFrac):
	#No. 1 < 2%, No. 2 < 6%, No. 3 < 20%
	if dgrFrac < 0.02:
		return 'No. 1'
	elif dgrFrac < 0.06:
		return 'No. 2'
	elif dgrFrac < 0.2:
		return 'No. 3'
	else:
		return 'No. 4'
	
	
	

#Define parameters
pgcPath = r"Images\PaleGreenChip.png"
dgrPath = r"Images\DGRChip.png"
#seedPaths = [r"Images\Seed4.png", r"Images\Seed9.png"	]
seedPaths = [r"Images\Seed{}.png".format(i+1) for i in range(9)]

#Read images
paleGreenChip = readImage(pgcPath)
DGRChip = readImage(dgrPath)
seedImgs = [readImage(seedPath) for seedPath in seedPaths]

#Average the DGR color
paleGreenChip = averageImgColor(paleGreenChip)
displayImg(paleGreenChip, title='averaged pg chip')

DGRChip = averageImgColor(DGRChip)
displayImg(DGRChip, title='averaged DGR chip')


for seedImg in seedImgs:
	displayImg(seedImg, title = 'Seed img')
	displayImg(convertBRGToHSV(seedImg), cmap='hsv', title = 'Seed img')

	#Compute watershed masks
	markerMsk = markBackForeGround(seedImg)
	watershedMsk = createWatershedMsk(seedImg, markerMsk)
	seedImg = maskImg(seedImg, watershedMsk, 1, background=False)
	displayImg(seedImg, title='masked img')
	seedImgFul = seedImg.copy()
	
	#Compute hsv distances
	colorDistances = differenceImgColor(seedImg, DGRChip)
	displayImg(colorDistances, cmap='gray', title = 'Image Differences')
	displayImg(colorDistances, cmap='jet', title = 'Image Differences')
	maskedDist = maskImg(seedImg, colorDistances, 210)
	displayImg(maskedDist, title='thresholded img')
	seedImgRel = maskImg(seedImg, colorDistances, 210, background=False)
	displayImg(seedImgRel, title='thresholded img')
	
	#Compute DGR distances
	colorDistances = differenceImgColor(seedImgRel.copy(), DGRChip, dgr='dgr')
	displayImg(colorDistances, cmap='gray', title = 'Image Differences') 
	displayImg(colorDistances, cmap='jet', title = 'Image Differences')
	maskedDist = maskImg(seedImgRel, colorDistances, 200)
	displayImg(maskedDist, title='thresholded img')
	seedImgDGR = maskImg(seedImgRel, colorDistances, 200, background=False)
	displayImg(seedImgDGR, title='thresholded img')

	#Display seed info
	displaySeedInfo(seedImgFul, seedImgRel, seedImgDGR)


