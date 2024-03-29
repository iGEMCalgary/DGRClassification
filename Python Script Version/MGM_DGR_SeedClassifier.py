#Imports
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import time
import os
from MGM_DGR_ImageFormatting import ImgUtility
from MGM_DGR_ImageFormatting import ImgCalibrator
from sklearn.preprocessing import MinMaxScaler

#Classifier for seeds grades them on how much they are DGR
class SeedClassifier:
	#Default constructor comes with an image formatter
	def __init__(self):
		#Parameters for calculations map
		#[0]=default value, [1]=actual value, [2]=limits
		self.paramMap={
			'bgdSatMax':		[35,	35,		(0,100)],
			'bgdNGSatMax':		[30,	30,		(0,100)],
			'bgdNGBriMin':		[25,	25,		(0,100)],
			'smrThreshMin':		[210,	210,	(0,255)],
			'smrDiffW':			[1.1,	1.1,	(0,10)],
			'smrGreyPenaltyW':	[1.4,	1.4,	(0,10)],
			'smrPxlMin':		[5,		5,		(0,900)],		
			'dgrThreshMin':		[100,	100, 	(0,255)],
			'dgrLRelMax':		[3.6,	3.6,	(0,100)],
			'dgrLMin':			[5,		5,		(0,100)],
			'dgrDistMax':		[20,	20,		(0,50)],
			'dgrAEdge':			[-0.42,	-0.42,	(-1.28, -0.4)],
			'dgrBEdge':			[0.62,	0.62, 	(0.4, 1.28)],
			'dgrFracMin':		[0.55,	0.55,	(0,1)]
		}

		self.sAnly = SeedSampleAnalyzer()

		self.settingsFileName = r'advancedSettings.txt'
		self.loadASettings()
		
		self.iForm = ImgUtility()
		self.sAnly.setConstants(self.paramMap['dgrFracMin'][1], self.paramMap['smrPxlMin'][1])
		
		self.colorChipCol = np.array([255,255,255])
	

	#get seed analyzer
	def getSeedAnly(self):
		return self.sAnly
	
	#determine the difference between the current time and the given start time
	def timeEndTime(self, startTime):
		endTime = time.time()
		deltaTime = endTime - startTime
		if deltaTime % 60 < 1:
			timeString = "Time: {:5.3f} milliseconds.".format((deltaTime%60)*1000)
		else:
			timeString = "Time: {} minutes, {:5.3f} seconds.".format(math.floor(deltaTime/60.0), deltaTime % 60)
		
		return timeString
	
	#Get the specific value from the analyzer parameter settings
	#values are 'default' for default parameters, 'actual' for actual parameters,
	#and 'limit' for the range of potential values
	def getASettings(self, key, value='actual'):
		valueDict = {'default':0, 'actual':1, 'limits':2}
		return self.paramMap[key][valueDict[value]]
	
	#set the actual values of all the parameters with a dictionary of all the settings
	#return true if set successful, return false if not
	def setASettings(self, paramA):
		#Copy all values in the dict over with all the keys
		if(paramA.keys()!=self.paramMap.keys()):	
			return False
		for key in self.paramMap.keys():
			if(paramA[key]<self.paramMap[key][2][0] or paramA[key]>self.paramMap[key][2][1]):
				return False
			self.paramMap[key][1]=paramA[key]
		
		#set settings for the seed analyzer
		self.sAnly.setConstants(self.paramMap['dgrFracMin'][1], self.paramMap['smrPxlMin'][1])
		
		#save the settings to a txt file
		settingsTxt = self.getASettingsString()
		settingsFile = open(self.settingsFileName, 'w')
		settingsFile.write(settingsTxt)
		settingsFile.close()
		
		return True
	
	#Load the parameter files if the advanced settings file exists.
	def loadASettings(self):
		if(not os.path.exists(self.settingsFileName) or not os.path.isfile(self.settingsFileName)):
			return
		settingsFile = open(self.settingsFileName, 'r')
		
		#parse settings from file
		count = 0
		for line in settingsFile:
			count = count+1			
			if count==1:			#skip first line
				continue
			words = line.split()
			self.paramMap[words[0][:-1]][1]=float(words[1])	
			
		settingsFile.close()
		self.sAnly.setConstants(self.paramMap['dgrFracMin'][1], self.paramMap['smrPxlMin'][1])
		
		print("Advanced Settings Loaded.")
	
	#Get the list of advanced settings as a string
	def getASettingsString(self):
		settingsTxt = 'Advanced Settings:\n'
		for key in self.paramMap.keys():
			settingsTxt = settingsTxt + key + ":			" + str(self.paramMap[key][1]) + "\n"
		return settingsTxt
		
	#reset the actual values to be the default parameters
	def resetASettings(self):
		for setting in self.paramMap.keys():
			self.paramMap[setting][1]=self.paramMap[setting][0]
		self.sAnly.setConstants(self.paramMap['dgrFracMin'][1], self.paramMap['smrPxlMin'][1])
	
	#Aqcuire the DGR ratings for each seed and return them
	def classifySeedSample(self, seedList, dgrColChip, displaySteps=False):
		classificationStartTime = time.time()			#Begin timing for the entire seed sample
		seedInfo = []									#List of seeds and their associated masked images and classification 		
		
		for seedImg in seedList:						#For each seed image:

			seedClassTime = time.time()					#Begin timing for each seed 			
			
			#Compute watershed masks
			seedImg, seedImgDisplayWM, seedImgFul = self.maskSeedFromBackgroundWithWatershed(seedImg)
			
			#Compute smear partition of seed imag using hsv distance threshhold methodology	
			seedImg, seedImgDisplaySI, seedImgRel = self.maskSmearFromSeedWithHSVDist(seedImg, dgrColChip)
			
			#Compute DGR distances and DGR pixels of the seed image
			seedImg, seedImgDisplayDG, seedImgDGR = self.maskDGRFromSmearWithLabDist(seedImg, dgrColChip)
			
			#Compute seed info
			seedFrac, seedAreas = self.sAnly.gradeSeed(seedImgFul, seedImgRel, seedImgDGR)
			if seedFrac != 'NaN':
				seedImgDisplayConf = np.full(seedImg.shape, self.sAnly.confidenceColour(seedFrac, seedAreas))
				seedInfo.append([(seedImgDisplayWM, seedImgDisplaySI, seedImgDisplayDG, seedImgDisplayConf), seedFrac, seedAreas])
			else:
				seedImgDisplayConf = np.full(seedImg.shape, self.sAnly.confidenceColour(-1, seedAreas))
				seedInfo.append([(seedImgDisplayWM, seedImgDisplaySI, seedImgDisplayDG, seedImgDisplayConf), -1, seedAreas])
			
			#Approximately 86ms
			print("Seed analyzed. " + self.timeEndTime(seedClassTime))
		
		timeString = "\nSeed sample classification finished. " + self.timeEndTime(classificationStartTime)
		print(timeString)
		
		return [seedInfo, timeString[38:]]
	
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
		notSaturated = imgPxl[1] < svConv(self.paramMap['bgdSatMax'][1])
		
		#exceptions exist for GREEN hue but almost saturated enough
		greenAlmostNotSaturated = greenHue(imgPxl[0]) and imgPxl[1] > svConv(self.paramMap['bgdNGSatMax'][1])
		
		#exceptions exist for GREEN hue and not saturated but very dark
		greenVeryDark = greenHue(imgPxl[0]) and imgPxl[2] < svConv(self.paramMap['bgdNGBriMin'][1])
		
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
		startTime = time.time()																	#Begin timing
		
		colorDistances = self.differenceImgColor(seedImg, DGRChip)								#Get the differences in each pixel as a grayscale image

		threshVal = self.paramMap['smrThreshMin'][1]											#Threshold value		
		seedImgDisplay = self.iForm.maskImg(seedImg, colorDistances, threshVal)					#Create the masked smear img
		seedImgRel = self.iForm.maskImg(seedImg, colorDistances, threshVal, background=False)	#Create the masked smear img for display
		seedImg = seedImgRel.copy()
		
		#print("Relevant Seed Determined. " + self.timeEndTime(startTime))						#Get time for smear speration process
		return [seedImg, seedImgDisplay, seedImgRel]
	
	#Return an image displaying the differences in color between a given img and a colour
	def differenceImgColor(self, imgP, colorChipP, dgr='rel'):
		
		if dgr=='rel':
			img = self.iForm.convertBGRToHSV(imgP)
			distanceType = lambda x: self.hueDistanceGreen(x)
		elif dgr=='dgr':
			img = self.iForm.convertBGRToLab(imgP)

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
				
		hueDiffWeight = self.paramMap['smrDiffW'][1]

		#Distance penalties for gray = 1.4(saturation-70/2)
		grayPenalty = lambda x: self.paramMap['smrGreyPenaltyW'][1]*(x-70/2) if x <= 70/2 else 0
				
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
			totalPenalty = totalPenalty + self.paramMap['dgrLRelMax'][1]*(LDiff**2)
		
		#Being too dark also incurs a penalty at L < 5
		#P = (-L+5)**2
		if pxl1[0] < self.paramMap['dgrLMin'][1]:	
			totalPenalty = totalPenalty + 2*(-pxl1[0]+5)**2
		
		#Color penalty maxes out at distance d=20 from ab region
		#P = rate*D = (90/d)*(limit-real)
		#a region is [-128, -0.42*L]
		rate = 90/self.paramMap['dgrDistMax'][1]	
		
		aDiff = pxl1[1] - self.paramMap['dgrAEdge'][1]*pxl1[0]
		if aDiff > 0:
			totalPenalty = totalPenalty + rate*aDiff
		
		#b region is [0, 0.62*L)]
		bDiff = self.paramMap['dgrBEdge'][1]*pxl1[0] - pxl1[2]
		if bDiff > 0:
			totalPenalty = totalPenalty + rate*bDiff
		
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

		threshVal = self.paramMap['dgrThreshMin'][1]											#Threshold value	
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
		self.dgrThreshold = 0.5		#dgr fraction must be greater than this
		self.smrPixelMin = 0		#relevant smear must be greater than this many pixels
		
		self.gradingColourDict = {'NotDGR':[0,255,255],'DGR':[0,0,255],'NaN':[255,0,0],
								  'cust_NotDGR':[19,128,128],'cust_DGR':[19,19,128],'cust_NaN':[128,19,19]}
		self.iForm = ImgUtility()
	
	#Set the constants of this analyzer
	def setConstants(self, dgrFracMin, smrPxlMin):
		self.dgrThreshold = dgrFracMin
		self.smrPixelMin = smrPxlMin
	
	#Grade an individual seed
	def gradeSeed(self, seedImgFul, seedImgRel, seedImgDGR):		
		#Count the number of non-zero ([0,0,0]) pixels in each image
		areaFul = areaRel = areaDGR = 0
		imgNonZero = lambda x: np.count_nonzero(np.mean(x, axis=2))
		areaFul = imgNonZero(seedImgFul)
		areaRel = imgNonZero(seedImgRel)
		areaDGR = imgNonZero(seedImgDGR)	
		
		analysisString = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
		
		#Display the information about the seed.
		if areaRel > self.smrPixelMin:
			dgrFrac = float(areaDGR)/float(areaRel)
			analysisString = analysisString + "The full area of the seed is: {} pixels.".format(areaFul)
			analysisString = analysisString + "The relevant/non-hull area of the seed is: {} pixels.".format(areaRel)
			analysisString = analysisString + "The DGR area of the seed is: {} pixels.".format(areaDGR)
			analysisString = analysisString + "The DGR % of the seed is: {:5.3f}% DGR".format(dgrFrac*100)
			analysisString = analysisString + "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
			print(analysisString)
		else:
			analysisString = analysisString + "No seed smear detected."
			analysisString = analysisString + "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
			print(analysisString)
			return ['NaN', (areaFul, areaRel, areaDGR)]
		
		return [dgrFrac, (areaFul, areaRel, areaDGR)]
		
	#Analyze a full sample
	def analyzeSeedSample(self, seedSampleInfo, timeString=None):
		#Count quantities of seeds
		totalSeedCount = len(seedSampleInfo) 
			
		usableSeedCount = max([[self.positive(seedInfo[1]) for seedInfo in seedSampleInfo].count(True), 1])
		greenSeedCount = [self.isDGR(seedInfo[1]) for seedInfo in seedSampleInfo].count(True)
		
		dgrFrac = greenSeedCount/usableSeedCount
		sampleGrade = self.gradeCanola(dgrFrac)
		
		analysisString = "\nThe grade of the Canola sample is: {} ({} DGR Seeds/{} Total Sample Seeds = {:5.3f}% DGR).".format(sampleGrade,greenSeedCount,usableSeedCount,dgrFrac*100)
		
		if totalSeedCount != greenSeedCount:
			analysisString = analysisString + " {} seed(s) were unable to be analyzed, and were thus removed from the grading process.".format(totalSeedCount-usableSeedCount)
	
		print(analysisString)
		
		if timeString == None:
			return [analysisString]
		else:
			return [analysisString, timeString]

	#Determine if a seed is usable or not (not 'NaN')
	def positive(self, dgrFrac):
		if dgrFrac == 'cust_DGR':
			return True
		elif dgrFrac == 'cust_NotDGR':
			return True 
		elif dgrFrac == 'cust_NaN':
			return False
		else:
			return dgrFrac >= 0

	#Determine if a seed is DGR or not based on its DGR pixel fraction
	def isDGR(self, dgrFrac):
		if dgrFrac == 'cust_DGR':
			return True
		elif dgrFrac == 'cust_NotDGR':
			return False 
		elif dgrFrac == 'cust_NaN':
			return False
		else:
			return dgrFrac > self.dgrThreshold

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
	
	#Give a BGR confidence colour from light to dark based on the given dgr fraction
	def confidenceColour(self, dgrFrac, areas):
		#Being close to the threshold decreases confidence
		#Formula is: Conf=500(dgrFrac-threshold)**2, maxing out at 100
		#Lower confidence comes from very low smear and dgr values
		#Formula is: Conf -= (areaDGR/areaRel)*10*np.log(-areaRel+40)+40
		if dgrFrac < 0:
			conf = 100
		else:
			conf = min([100, ((dgrFrac-self.dgrThreshold)**2)*500]) 
			if areas[1] < 40:
				negConf = max([0, (10*np.log(-areas[1]+40)+40)*(areas[2]/areas[1])])
				conf = max([0, conf-negConf])
			
		#Translate confidence from 0 to 100 as brightness
		#R = 0
		#G = 255/100*Conf
		#B = 48/100 *Conf
		return [0.48*conf, 2.55*conf, 0]
	
	#Create composite images for display of relevant and dgr seeds in a sample
	def createInfoImages(self, seedSampleInfo, rows, cols):
		
		#Overlay the grids
		displaySeedSampleInfo = [(self.borderClassify(seedInfo[0],seedInfo[1]),seedInfo[1]) for seedInfo in seedSampleInfo]
		
		#Assemble the wm images, then the rel images, then the dgr images
		wmImgList = [seedInfo[0][0] for seedInfo in displaySeedSampleInfo]
		relImgList = [seedInfo[0][1] for seedInfo in displaySeedSampleInfo]
		dgrImgList = [seedInfo[0][2] for seedInfo in displaySeedSampleInfo]
		confImgList = [seedInfo[0][3] for seedInfo in displaySeedSampleInfo]

		#Assemble each row
		wmImgRows = []
		relImgRows = []
		dgrImgRows = []
		confImgRows = []
	
		for i in range(rows):
			wmImgRows.append(np.hstack(tuple(wmImgList[i*cols:(i+1)*cols])))
			relImgRows.append(np.hstack(tuple(relImgList[i*cols:(i+1)*cols])))
			dgrImgRows.append(np.hstack(tuple(dgrImgList[i*cols:(i+1)*cols])))
			confImgRows.append(np.hstack(tuple(confImgList[i*cols:(i+1)*cols])))
			
		#Combine the rows into the full image 
		wmImg = np.vstack(tuple(wmImgRows))
		relImg = np.vstack(tuple(relImgRows))
		dgrImg = np.vstack(tuple(dgrImgRows))
		confImg = np.vstack(tuple(confImgRows))
				
		return [wmImg, relImg, dgrImg, confImg]
		
	#Mark each given image with an appropriately colored two-pixel thick border
	def borderClassify(self, seedImgTrio, seedFrac):
		#Determine appropriate border color
		if seedFrac in self.gradingColourDict:
			borderCol = self.gradingColourDict[seedFrac]
		elif seedFrac < 0:
			borderCol = self.gradingColourDict['NaN']
		elif self.isDGR(seedFrac):
			borderCol = self.gradingColourDict['DGR']
		else:
			borderCol = self.gradingColourDict['NotDGR']

		seedBorderedList = [
		#self.iForm.drawBox(
		#self.iForm.drawBox(
		self.iForm.drawBox(
		self.iForm.drawBox(
		self.iForm.drawBox(seedImg, (0,0), (seedImg.shape[1]-1,seedImg.shape[0]-1), overlayCol=borderCol),
		(1,1), (seedImg.shape[1]-2,seedImg.shape[0]-2), overlayCol=borderCol), 
		(2,2), (seedImg.shape[1]-3,seedImg.shape[0]-3), overlayCol=borderCol) for seedImg in seedImgTrio]
		#(3,3), (seedImg.shape[1]-4,seedImg.shape[0]-4), overlayCol=borderCol), 
		#(4,4), (seedImg.shape[1]-5,seedImg.shape[0]-5), overlayCol=borderCol) for seedImg in seedImgTrio]
				
		return tuple(seedBorderedList)
		
		
			