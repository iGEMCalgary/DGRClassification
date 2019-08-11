#Imports
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import queue
import os


#Provides basic utility functions for images
class ImgUtility:		
	#Display the given image
	def displayImg(self, img, cmap='bgr', title='', block=True):
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
	
	#Read and return a specific image from a given path
	def readImageFromFolder(self, folderPath, imgPath):
		return self.readImage(str(os.path.join(folderPath, imgPath)))
	
	#Save a specific image to a given path in a folder
	def saveImageToFolder(self, folderPath, imgPath, img):
		cv2.imwrite(str(os.path.join(folderPath, imgPath)), img)
	
	#Read and return the most recent image in the folder given
	def readLatestImageFromFolder(self, folderPath, startOffset=0):
		sortedFiles = self.sortedFilesInFolder(folderPath)
		sortedFiles.reverse()
		for filePath in sortedFiles[startOffset:]:
			if filePath[-4:]=='.png' or filePath[-4:]=='jpg':
				print("Read image file: " + filePath)
				return self.readImage(str(os.path.join(folderPath,filePath)))
		
	#Returns a list of file paths given a folder path, sorted by timestamps
	def sortedFilesInFolder(self, directoryPath):
		mTime = lambda f: os.stat(os.path.join(directoryPath, f)).st_mtime
		return list(sorted(os.listdir(directoryPath), key=mTime))
	
	#Read and return the 3-channel BGR image at the path given.
	def readImage(self, path):
		return cv2.imread(path, cv2.IMREAD_COLOR)
	
	#Convert a BGR image into a hsv img
	def convertBGRToHSV(self, img):
		#Converting from HSV in plt to real HSV:
		#H = plt.H * 2
		#S = plt.S / 2.55
		#V = plt.V / 2.55
		return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	#Convert an HSV image into a BGR image
	def convertHSVToBGR(self, img):
		return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
	
	#Convert a BGR image into a CIE L*a*b image
	def convertBGRToLab(self, img):
		#Converting from Lab in plt to real Lab:
		#L = plt.L / 2.55
		#a = plt.a + 128
		#b = plt.b + 128
		return cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
	
	#Convert a Lab image into a BGR image
	def convertLabToBGR(self, img):
		return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
	
	#Convert a BGR image into a merged RGB array
	def convertNpToRGBArr(self, img):
		img = img.astype(np.uint8)
		b,g,r = cv2.split(img)
		return cv2.merge((r,g,b))
	
	#Crop an image. Can use either fractions or pixel values.
	def cropImg(self, img, leftTopCorner, rightBotCorner):
		if float(leftTopCorner[0]) < 1 or float(rightBotCorner[0]) < 1:
			leftTopCorner = (float(leftTopCorner[0]),float(leftTopCorner[1]))
			rightBotCorner = (float(rightBotCorner[0]),float(rightBotCorner[1]))
			return img[round(leftTopCorner[1]*img.shape[0]):round(rightBotCorner[1]*img.shape[0]),round(leftTopCorner[0]*img.shape[1]):round(rightBotCorner[0]*img.shape[1])] 
		else:
			leftTopCorner = (int(leftTopCorner[0]),int(leftTopCorner[1]))
			rightBotCorner = (int(rightBotCorner[0]),int(rightBotCorner[1]))
			return img[leftTopCorner[1]:rightBotCorner[1], leftTopCorner[0]:rightBotCorner[0]]
	
	#Resize an image.
	def resizeImg(self, img, xF, yF):
		resizedImg = cv2.resize(img, (int(xF),int(yF))).astype(np.uint8)
		return resizedImg
	
	#Turn an image (copy) into a list of subimages, according to given grid parameters.
	def gridPartitionImg(self, img, numRows, numCols):
		cellImgList = []
		w = img.shape[1]/numCols		#w = width/(# of partitions horizontal)
		h = img.shape[0]/numRows		#h = height/(# of partitions vertical)
		y=x=0
		while math.ceil(y) < img.shape[0]:
			while math.ceil(x) < img.shape[1]:
				cellImgList.append(img[round(y):round(y+h),round(x):round(x+w)])
				x = x+w
			x = 0
			y = y+h

		return cellImgList
	
	#Draw a (teal) rectangle on an image
	#xRange and yRange are tuples of coordinates, inclusive
	def drawLine(self, img, xRange, yRange, overlay):
		tempImg = img.copy()
		tempImg[yRange[0]:yRange[1]+1,xRange[0]:xRange[1]+1,:] = overlay[yRange[0]:yRange[1]+1,xRange[0]:xRange[1]+1,:]
		return tempImg
	
	#Draw a (teal) grid on an image
	def drawGrid(self, img, leftTopCornerP, rightBotCornerP, rows, cols, overlayCol=[255,255,0]):
		#Create solid raw overlay 
		overlay=np.full(img.shape, np.array(overlayCol), dtype=np.uint8)
		imgTemp = img.copy()
		
		#Format coordinates if in fraction form
		if leftTopCornerP[0]<1. or leftTopCornerP[1]<1. or rightBotCornerP[0]<1. or rightBotCornerP[1]<1.:
			leftTopCorner = (int(leftTopCornerP[0]*img.shape[1]),int(leftTopCornerP[1]*img.shape[0]))
			rightBotCorner = (int(rightBotCornerP[0]*img.shape[1]),int(rightBotCornerP[1]*img.shape[0]))
		else:
			leftTopCorner = leftTopCornerP.copy()
			rightBotCorner = rightBotCornerP.copy()
		
		#Calculate seperations
		rowSep = int((rightBotCorner[1] - leftTopCorner[1]) / rows)
		colSep = int((rightBotCorner[0] - leftTopCorner[0]) / cols)
		
		#Draw rows if rows>1
		if rows>1:
			for i in range(rows+1):
				imgTemp = self.drawLine(imgTemp,(leftTopCorner[0],rightBotCorner[0]),
										   (leftTopCorner[1]+i*rowSep,leftTopCorner[1]+i*rowSep),overlay=overlay)
			
		#Draw columns if cols>1
		if cols>1:
			for i in range(cols+1):
				imgTemp = self.drawLine(imgTemp,(leftTopCorner[0]+i*colSep,leftTopCorner[0]+i*colSep),
										   (leftTopCorner[1],rightBotCorner[1]),overlay=overlay)
		
		return imgTemp
	
	#Draw a (teal) crop box on an image
	def drawBox(self, img, leftTopCornerP, rightBotCornerP, overlayCol=[255,255,0]):
		#Create solid raw overlay 
		overlay=np.full(img.shape, np.array(overlayCol), dtype=np.uint8)
		imgTemp = img.copy()
		
		#Format coordinates if in fraction form
		if rightBotCornerP[0]<=1. and rightBotCornerP[1]<=1.:
			leftTopCorner = (int(leftTopCornerP[0]*img.shape[1]),int(leftTopCornerP[1]*img.shape[0]))
			rightBotCorner = (int(rightBotCornerP[0]*img.shape[1]),int(rightBotCornerP[1]*img.shape[0]))
		else:
			leftTopCorner = leftTopCornerP
			rightBotCorner = rightBotCornerP
		
		#Draw upper and lower rows
		imgTemp = self.drawLine(imgTemp,(leftTopCorner[0],rightBotCorner[0]),(leftTopCorner[1],leftTopCorner[1]),overlay=overlay)
		imgTemp = self.drawLine(imgTemp,(leftTopCorner[0],rightBotCorner[0]),(rightBotCorner[1],rightBotCorner[1]),overlay=overlay)		
		
		#Draw left and right columns
		imgTemp = self.drawLine(imgTemp,(leftTopCorner[0],leftTopCorner[0]),(leftTopCorner[1],rightBotCorner[1]),overlay=overlay)
		imgTemp = self.drawLine(imgTemp,(rightBotCorner[0],rightBotCorner[0]),(leftTopCorner[1],rightBotCorner[1]),overlay=overlay)		
		
		return imgTemp
		
	#Turn an image (copy) into a solid image of averaged color
	def averageImg(self, img):
		average = img.mean(axis=0).mean(axis=0)
		averageImg = np.ones(shape=img.shape, dtype=np.uint8) * np.uint8(average)
		return averageImg
		
	#Create a mask using thresholds
	def maskImg(self, img, msk, threshold, background=True):
		blurredImg = np.zeros_like(img)
		if background == True:
			blurredImg = cv2.blur(img.copy(), (5,5))
			for i in range(1):
				blurredImg = cv2.blur(blurredImg, (5,5))
			blurredImg = self.adjustGamma(blurredImg, gamma = 0.6)

		if len(msk.shape) == 2:
			maskedImg = blurredImg.copy()
			maskedImg[msk >= threshold] = img[msk >= threshold]
		else:
			maskedImg = np.where(msk >= threshold, img, blurredImg)

		return maskedImg
	
	#Adjust the gamme value of an image
	def adjustGamma(self, img, gamma=1.0):

	   invGamma = 1.0 / gamma
	   table = np.array([((i / 255.0) ** invGamma) * 255
		  for i in np.arange(0, 256)]).astype("uint8")

	   return cv2.LUT(img, table)
		
	#Fill in small holes (val <= 0)in the image.
	#maxSize = 85
	def fillHoles(self, img, maxSize = 524):
		#startTime = time.time()
		workImg = img.copy()
		#Use floodfill and fill in the image
		i = j = 1
		while i < img.shape[0]-1:
			while j < img.shape[1]-1:
				if self.fillCriteria(img, i, j):
					img, workImg = self.floodfill(img, workImg, i, j, maxSize)
				self.removeSinglePixel(img, i, j, 255)
				j = j+1
				
			i = i+1
			j = 1
		
		return img

	#Return true if the current pixel should be filled in.
	#returns false if the pxl's cross area is all blank.
	def fillCriteria(self, img, i, j, threshold=0):
		return img[i][j]<=threshold and (img[i-1][j]>threshold or img[i+1][j]>threshold or img[i][j-1]>threshold or img[i][j+1]>threshold) 

	#FloodFill algorithm that reverts changes if the flooded blob is too large
	#Uses a modified scanline algorithm.
	def floodfill(self, img, workImg, i, j, maxSize, replacedC = 0, replacementC = 255):
		
		#Floodfill algorithm
		if(i < 0 or j < 0 or i == img.shape[0] or j == img.shape[1] or img[i][j] > replacedC):
			return img
		
		#Markers for covering examined pixels must be higher in value than the
		#replacement value, and success or failure values must differ from each other.
		initRepC = replacementC + 100
		finaRepC = replacementC + 50
		
		#imgC = img.copy()	
		#imgC[i][j] = replacementC
		workImg[i][j] = initRepC
		
		Q = queue.Queue()	
		Q.put((i,j))
		
		lengthCounter = 0
		
		while not Q.empty():
			n = Q.get()

			#Do not keep the fill if the fill is too large or if it is touching another failed fill
			lengthCounter = lengthCounter + 1
			if lengthCounter > maxSize or self.touchingColour(n, workImg, finaRepC):
				workImg[workImg==initRepC] = finaRepC
				return [img, workImg]
			
			#west
			if(n[1] > 0 and workImg[n[0]][n[1]-1] <= replacedC):
				workImg[n[0]][n[1]-1] = initRepC
				Q.put((n[0],n[1]-1))
			#east
			if(n[1] < img.shape[1] - 2 and workImg[n[0]][n[1]+1] <= replacedC):
				workImg[n[0]][n[1]+1] = initRepC
				Q.put((n[0],n[1]+1))
			#north
			if(n[0] > 0 and workImg[n[0]-1][n[1]] <= replacedC):
				workImg[n[0]-1][n[1]] = initRepC
				Q.put((n[0]-1,n[1]))
			#south
			if(n[0] < img.shape[0]-2 and workImg[n[0]+1][n[1]] <= replacedC):
				workImg[n[0]+1][n[1]] = initRepC
				Q.put((n[0]+1,n[1]))
		
		img[workImg==initRepC] = replacementC
		workImg[workImg==initRepC] = finaRepC	
		return [img, workImg]
		
	#Checks if a given pixel in a grayscale image is adjacent to another pixel with the given value
	def touchingColour(self, loc, img, touchCol):
		#west
		if(loc[1] > 0 and img[loc[0]][loc[1]-1] == touchCol):
			return True
		#east
		if(loc[1] < img.shape[1] - 2 and img[loc[0]][loc[1]+1] == touchCol):
			return True
		#north
		if(loc[0] > 0 and img[loc[0]-1][loc[1]] == touchCol):
			return True
		#south
		if(loc[0] < img.shape[0]-2 and img[loc[0]+1][loc[1]] == touchCol):
			return True
		return False
		
	#Removes a pixel that is solitary from an image
	def removeSinglePixel(self, img, i, j, threshold):
		if img[i][j] >= threshold and img[i-1][j]<threshold and img[i+1][j]<threshold and img[i][j-1]<threshold and img[i][j+1]<threshold:
			img[i][j] = 0
			
#Class for taking pictures
class ImgCamera:		
	#Constructor gets camera index, defaults to webcam if available
	#If cameraIndex = None, attempts to search for the camera (webcam last)
	def __init__(self, cameraIndex = 0):
		#The default saved image is just a black 200x200 image.
		self.memImg = np.zeros((200,200,3),dtype=np.uint8)
		
		if cameraIndex == None:
			cameraIndexPossibleRange = 200
			cameraFound = False 
			for i in range(cameraIndexPossibleRange)[1:]:
				if takePicture(custCamIndex=i, wantStatus=True):
					print("Camera located at index {}".format(i))
					self.cameraIndex = i
					cameraFound = True
			if not cameraFound:
				self.cameraIndex = 0
		else:
			self.cameraIndex = cameraIndex
	
	#Take a picture using the camera and return the status or image
	#Also save the most recent image.
	def takePhoto(self, custCamIndex = None, wantStatus=False):
		if custCamIndex == None:
			camInd = self.cameraIndex
		else:
			camInd = custCamIndex
			
		camera = cv2.VideoCapture(camInd)
		status, camImg = camera.read()
		if wantStatus:
			return status
		elif status:
			self.memImg = camImg
			return camImg
			
	#Get the most recent image taken
	def getMostRecentPhoto(self):
		return self.memImg

#Provides colour calibration for images
class ImgCalibrator:
	#Default constructor
	def __init__(self, imgCamera, saveFolderPath, saveFilePath):
		#Get image formatter
		self.iForm = ImgUtility()
		self.iCamr = imgCamera
		
		#Set the calibration image save path
		self.saveFolderPath = saveFolderPath
		self.saveFilePath = saveFilePath
		
		#Set the default offset of no change
		self.offset = [0,0,0]
		
		#Record of true colour HSV values
		self.colourDict = {'white':[0,0,255]}
	
	#Take and save a calibration photo
	def takeCalibrationPhoto(self, img=[], leftTopCorner=(0.,0.), rightBotCorner=(1.,1.), filePath=None):
		if filePath==None and len(img)==0:
			calibPhotoOG = self.iCamr.takePhoto()
		elif filePath != None:
			calibPhotoOG = self.iForm.readImage(filePath)
		else:
			calibPhotoOG = img
		calibPhoto = self.iForm.averageImg(self.iForm.cropImg(calibPhotoOG, leftTopCorner, rightBotCorner))
		self.iForm.saveImageToFolder(self.saveFolderPath, self.saveFilePath, calibPhoto)
		return (calibPhoto, calibPhotoOG)
	
	#Calibrate a colour in BGR to BGR
	def calibrateCol(self, img, colour='white'):
		#average the image for calibration use
		aveImg = self.iForm.averageImg(img)
		aveImg = self.iForm.convertBRGToHSV(aveImg)
		
		#Calculate the offset from the true value
		self.offset = [self.colourDict[colour][i] - aveImg[0][0][i] for i in range(3)]
		self.offset[0] = 0
		
	#Apply the current offset to the image
	def correctImg(self, img):
		#Correct the img
		offsetImg = np.full(img.shape, self.offset)
		correctedImg = self.iForm.convertBRGToHSV(img) + offsetImg
		correctedImg = correctedImg.astype(np.uint8)
		correctedImg = self.iForm.convertHSVToBGR(correctedImg)
		
		#Normalize the img
		limitImg = np.full(img.shape, self.colourDict['white'])
		limitImg = self.iForm.convertHSVToBGR(limitImg.astype(np.uint8))
		correctedImg = np.where(correctedImg<=255, correctedImg, limitImg)
		
		return correctedImg
		
		
		
		
		
		
		