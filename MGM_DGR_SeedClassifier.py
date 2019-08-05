#Imports
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import time
from MGM_DGR_ImageFormatting import ImgUtility
from MGM_DGR_ImageFormatting import ImgCalibrator
from sklearn.preprocessing import MinMaxScaler

#Classifier for seeds grades them on how much they are DGR
class SeedClassifier:
	#Default constructor comes with an image formatter
	def __init__(self):
		self.iForm = ImgUtility()
		self.sAnly = SeedSampleAnalyzer()
		self.colorChipCol = np.array([255,255,255])
		
	#determine the difference between the current time and the given start time
	def timeEndTime(self, startTime):
		endTime = time.time()
		deltaTime = endTime - startTime
		if deltaTime % 60 < 1:
			timeString = "Time: {:5.3f} milliseconds.".format((deltaTime%60)*1000)
		else:
			timeString = "Time: {} minutes, {:5.3f} seconds.".format(math.floor(deltaTime/60.0), deltaTime % 60)
		
		return timeString
	
	#Aqcuire the DGR ratings for each seed and return them
	def classifySeedSample(self, seedList, dgrColChip, displaySteps=False):
		classificationStartTime = time.time()			#Begin timing for the entire seed sample
		seedInfo = []									#List of seeds and their associated masked images and classification 		
		
		for seedImg in seedList:						#For each seed image:

			seedClassTime = time.time()					#Begin timing for each seed 
			#iForm.displayImg(seedImg, title = 'Seed img')		#Display the original seed image
			
			#Compute watershed masks
			seedImg, seedImgDisplayWM, seedImgFul = self.maskSeedFromBackgroundWithWatershed(seedImg)
			
			#Compute smear partition of seed imag using hsv distance threshhold methodology	
			seedImg, seedImgDisplaySI, seedImgRel = self.maskSmearFromSeedWithHSVDist(seedImg, dgrColChip)
			
			#Compute DGR distances and DGR pixels of the seed image
			seedImg, seedImgDisplayDG, seedImgDGR = self.maskDGRFromSmearWithLabDist(seedImg, dgrColChip)
			
			#Compute seed info
			seedFrac = self.sAnly.gradeSeed(seedImgFul, seedImgRel, seedImgDGR)
			if seedFrac != 'NaN':
				seedInfo.append((seedImgDisplayWM, seedFrac))
			
			self.iForm.displayImg(seedImgDisplayWM)
			self.iForm.displayImg(seedImgRel)
			self.iForm.displayImg(seedImgDGR)
			
			#Approximately 86ms
			print("Seed analyzed. " + self.timeEndTime(seedClassTime))

		print("\nSeed sample classification finished. " + self.timeEndTime(classificationStartTime) + "\n")
		
		return seedInfo
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Watershed~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
	#Mask the seed from the base image using watershed
	#Returns [masked img, masked img for display, full seed img]
	def maskSeedFromBackgroundWithWatershed(self, seedImg):
		watershedTime = time.time()										#Begin timing for the watershed mask creation
		
		markerMsk = self.markBackForeGround(seedImg)							#Primary background-foreground marking
		watershedMsk = self.createWatershedMsk(seedImg, markerMsk)			#Create watershed mask
		
		seedImgDisplay = self.iForm.maskImg(seedImg, watershedMsk, 1)				#create a masked image for display
		seedImg = self.iForm.maskImg(seedImg, watershedMsk, 1, background=False)	#mask the image
		
		seedImgFul = seedImg.copy()													#create the pure seed copy

		#print("Watershed Mask created. " + self.timeEndTime(watershedTime))	#Get time for watershed mask creation
		return [seedImg, seedImgDisplay, seedImgFul]

	#Determines if a given HSV pixel is background.
	def isBackground(self, imgPxl):
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
		
		return notSaturated and not greenAlmostNotSaturated and not greenVeryDark
	
	#Mark background and foreground pixels in the image
	def markBackForeGround(self, img):		
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
					if self.isBackground(imgHSV[i][j]):
						#Background marker
						markerMsk[i][j] = 1
					else:
						# Dictate the area of interest		
						#Each tag that hits a smear gets a smear
						markerMsk[i][j] = 255
					cooldownCur = cooldownCap
				cooldownCur = cooldownCur - 1
		
		return markerMsk
	
	#Create BW watershed mask 
	def createWatershedMsk(self, img, markerMsk):		
		markerMskCopy = markerMsk.copy()							#Copy the initial marker mask
		
		#Initial watershed mask estimate:
		#Make the background black, and what we want to keep white
		watershedMsk = cv2.watershed(img, markerMskCopy)			

		watershedMsk[watershedMsk == 1] = 0							#Make the mask binary
		watershedMsk[watershedMsk > 1] = 255
		
		watershedMsk = self.iForm.fillHoles(watershedMsk.copy())	#Fill the holes of the initial mask
		
		return watershedMsk
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Smear~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
	#Mask the smear from the seed image using hsv hardcoded threshold values
	#Returns [masked img, masked img for display, full smear img]
	def maskSmearFromSeedWithHSVDist(self, seedImg, DGRChip):
		startTime = time.time()														#Begin timing
		
		colorDistances = self.differenceImgColor(seedImg, DGRChip)						#Get the differences in each pixel as a grayscale image

		threshVal = 210																#Threshold value		
		seedImgDisplay = self.iForm.maskImg(seedImg, colorDistances, threshVal)				#Create the masked smear img
		seedImgRel = self.iForm.maskImg(seedImg, colorDistances, threshVal, background=False)	#Create the masked smear img for display
		seedImg = seedImgRel.copy()
		
		#print("Relevant Seed Determined. " + self.timeEndTime(startTime))				#Get time for smear speration process
		return [seedImg, seedImgDisplay, seedImgRel]
	
	#Return an image displaying the differences in color between a given img and a colour
	def differenceImgColor(self, imgP, colorChipP, dgr='rel'):
		
		if dgr=='rel':
			img = self.iForm.convertBGRToHSV(imgP)
			distanceType = lambda x: self.hueDistanceGreen(x)
		elif dgr=='dgr':
			img = self.iForm.convertBGRToLab(imgP)
			self.iForm.displayImg(img, cmap=None)
			colorChip = self.iForm.convertBGRToLab(colorChipP)
			self.colorChipCol = [colorChip[0][0][0]/2.55, colorChip[0][0][0]-128, colorChip[0][0][0]-128]
			distanceType = lambda x: self.labDistanceChip(x)
		
		#Compute distances in colour
		diffImg = np.ones(shape=img.shape[:2])
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				diffImg[i][j] = distanceType(img[i][j])
							
		#Scale and format the colour distances appropriately
		diffImgCopy = np.zeros_like(diffImg)
		diffImgCopy[0] = np.array([0 for i in range(diffImgCopy.shape[1])])
		diffImgCopy[1] = np.array([90 for i in range(diffImgCopy.shape[1])])
			
		diffScaler = MinMaxScaler()
		diffScaler.fit(diffImgCopy)	
		
		diffImg = diffScaler.transform(diffImg)
	
		diffImg = diffImg * 255
		diffImg = diffImg.astype(np.uint8)
		diffImg = self.reverseImg(diffImg)
		
		return diffImg
	
	#Compute differences in hue, saturation, and value	
	def hueDistanceGreen(self, pxl1):

		#convert pixels to float channels
		pxl1 = [float(pxl1[0]), float(pxl1[1]), float(pxl1[2])]
		
		if pxl1[0]==0 and pxl1[1]==0 and pxl1[2]==0:
			return 90
				
		hueDiffWeight = 1.1

		#Distance penalties for gray = 1.4(x-70/2)
		grayPenalty = lambda x: 1.4*(x-70/2) if x <= 70/2 else 0
				
		if pxl1[0] > 70/2 and pxl1[0] <= 130/2:
			return grayPenalty(pxl1[1])
		elif pxl1[0] <= 70/2:
			return min([hueDiffWeight*abs(pxl1[0] - 70/2), hueDiffWeight*abs(180-(pxl1[0] - 70/2))]) + grayPenalty(pxl1[1])
		else:
			return min([hueDiffWeight*abs(pxl1[0] - 130/2), hueDiffWeight*abs(180-(pxl1[0] - 130/2))]) + grayPenalty(pxl1[1])
		
	#Compute differences between a pixel and a color in Lightness, a, and b
	def labDistanceChip(self, pxl1):
		#Convert pixels to Lab real values
		pxl1 = [float(pxl1[0])/2.55, float(pxl1[1])-128., float(pxl1[2])-128.]
		if pxl1[0]==0 and pxl1[1]==0 and pxl1[2]==0:
			return 90
		
		#Distance penalties for being lighter than the colorChipCol is severe.
		#Distance penalties for straying from the region of green is also severe.
		totalPenalty = 0
		
		#Lighter penalty maxes out at +5 L
		#P = 3.6*(1.L-2.L)**2
		LDiff = pxl1[0] - self.colorChipCol[0]
		if LDiff > 0:
			totalPenalty = totalPenalty + 3.6*(LDiff**2)
		
		#Color penalty maxes out at distance d=20 from ab region
		#P = rate*D = (90/d)*(limit-real)
		#a region is [-128, -0.42*L]
		rate = 90/20
		
		aDiff = pxl1[1] - (-0.42)*pxl1[0]
		if aDiff > 0:
			totalPenalty = totalPenalty + rate*aDiff
		
		#b region is [0, 0.62*L)]
		bDiff = 0.62*pxl1[0] - pxl1[2]
		if bDiff > 0:
			totalPenalty = totalPenalty + rate*bDiff
		
		#if totalPenalty < 100:
		#	print("lol")
		#print(totalPenalty)
		
		#Max penalty is 90
		return min([totalPenalty, 90])
		
	#Make all white black and black white
	def reverseImg(self, diffImg):
		return -1 * diffImg + 255 	
		
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DGR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
	#Mask the DGR pixels from the smear image using hsv hardcoded threshold values
	#Returns [masked img, masked img for display, full smear img]
	def maskDGRFromSmearWithLabDist(self, seedImg, DGRChip):
		startTime = time.time()																	#Begin timing
		
		colorDistances = self.differenceImgColor(seedImg.copy(), DGRChip, dgr='dgr')			#Get the differences in each pixel as a grayscale image

		self.iForm.displayImg(colorDistances, cmap='gray')


		threshVal = 100																			#Threshold value	
		seedImgDisplay = self.iForm.maskImg(seedImg, colorDistances, threshVal)					#Create the masked dgr img
		seedImgDGR = self.iForm.maskImg(seedImg, colorDistances, threshVal, background=False)	#Create the masked dgr img for display
		seedImg = seedImgDGR.copy()
		
		#print("DGR pixels determined." + self.timeEndTime(startTime))							#Determine timing
		return [seedImg, seedImgDisplay, seedImgDGR]
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
#Analyzes seed sample statistics
class SeedSampleAnalyzer:
	#Default constructor
	def __init__(self):
		self.x=5
		
	#Grade an individual seed
	def gradeSeed(self, seedImgFul, seedImgRel, seedImgDGR):		
		#Count the number of non-zero ([0,0,0]) pixels in each image
		areaFul = areaRel = areaDGR = 0
		imgNonZero = lambda x: np.count_nonzero(np.mean(x, axis=2))
		areaFul = imgNonZero(seedImgFul)
		areaRel = imgNonZero(seedImgRel)
		areaDGR = imgNonZero(seedImgDGR)
		
		#Display the information about the seed.
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
			return 'NaN'
			print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		
		return dgrFrac
		
	#Analyze a full sample
	def analyzeSeedSample(self, seedSampleInfo, totalSeedCount):
		usableSeedCount = len(seedSampleInfo) 
		greenSeedCount = [self.isDGR(seedInfo[1]) for seedInfo in seedSampleInfo].count(True)
		
		dgrFrac = greenSeedCount/usableSeedCount
		sampleGrade = self.gradeCanola(dgrFrac)
		print("The grade of the Canola sample is: {} ({} DGR Seeds/{} Total Sample Seeds = {}% DGR).".format(sampleGrade,greenSeedCount,usableSeedCount,dgrFrac*100))
		if totalSeedCount != greenSeedCount:
			print("{} seeds were unable to be analyzed, and were thus removed from the grading process.".format(totalSeedCount-usableSeedCount))
	
	#Determine if a seed is DGR or not based on its DGR pixel fraction
	def isDGR(self, dgrFrac):
		return dgrFrac > 0.5
	
	#Return the grade of canola based on fraction of dgr seeds.
	def gradeCanola(self, dgrFrac):
		#No. 1 < 2%, No. 2 < 6%, No. 3 < 20%
		if dgrFrac < 0.02:
			return 'No. 1'
		elif dgrFrac < 0.06:
			return 'No. 2'
		elif dgrFrac < 0.2:
			return 'No. 3'
		else:
			return 'No. 4'
			