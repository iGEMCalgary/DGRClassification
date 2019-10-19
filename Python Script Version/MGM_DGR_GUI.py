#Imports
import sys
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import copy
import re
import time 

#Import GUI
from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk

#Import Backend
from MGM_DGR_ImageFormatting import ImgUtility
from MGM_DGR_ImageFormatting import ImgCalibrator
from MGM_DGR_ImageFormatting import ImgCamera
from MGM_DGR_SeedClassifier import SeedClassifier

#~~~~~~~~~~~~~~~~~~~~~~~Create Backend~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Other classes 
iForm = ImgUtility()
iCamr = ImgCamera(cameraIndex=None)
iCali = ImgCalibrator(iCamr, "CalibrationImages", "guiCalibrationPhoto.png")
sClas = SeedClassifier()

#Convert an image from numpy format to GUI usable format
convImg = lambda x: ImageTk.PhotoImage(image=Image.fromarray(iForm.convertNpToRGBArr(x)))

#Create a black 3 channel RGB numpy image with the given dimensions (x,y)
blankImg = lambda x: np.zeros((x[1], x[0], 3))

#Browse for an image file path
browseImageFromComp = lambda x: filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select preview image file",
														filetypes = (("all Files","*.*"), ("jpeg Files","*.jpg"), ("png Files", "*.png"), 
																	 ("ppm Files", "*.ppm"), ("pgm Files", "*.pgm"), ))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~Create FrontEnd~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class GUI(Frame):
	def __init__(self, master=None):
		#Basic frame creation with fixed width and height range
		Frame.__init__(self, master)
		master.title("yOIL GreatGrader: Seed Sample Classifier")
		wMin,hMin = 1100, 600
		wMax, hMax = 1280, 720
		master.minsize(width=wMin, height=hMin)
		master.maxsize(width=wMax, height=hMax)
		self.pack()
		
		"""
		Left Button Panel
		4 buttons, with labels
		preview 	= take preview image. May be changed to a toggle of camera live feed in future.
		calibrate 	= take (and display) calibration photo.
		sample		= take (and display) seed sample photo. Do not grade yet.
		grade 		= grade (and display) graded photo with contrast or something.
		confidence 	= toggle displaying confidence heatmap. May be changed to allow for more images to be displayed.
		
		grade is dependant on there being a calibration and sample picture taken.
		confidence is dependant on there being a grade done.
		"""
		#Create button canvas panel
		
		buttonPanel = Canvas(master, width=200, height=hMin, bg="#8ac926")
		buttonPanel.pack(side=LEFT)
		
		#Create buttons to activate phone camera through IP Webcam
		self.ipCamBut = Button(buttonPanel, text="Activate IP Webcam Phone Camera", command=self.ipCamButCmd, bg="#829e83").grid(row=0, column=0, columnspan=4, pady=2)
		
		self.ipCamIPStrVar = StringVar()
		self.ipCamIP = Entry(buttonPanel, textvariable=self.ipCamIPStrVar, width=17).grid(sticky='W', row=1, column=2, columnspan=3)
		self.ipCamIPLab = Label(buttonPanel, text="ip: http://", bg="#829e83").grid(sticky='E', row=1,column=1)

		#Create the buttons and their labels
		self.preButL = Button(buttonPanel, text='Pc', width=2, command=self.preButCmd, bg="#1982C4").grid(row=2,column=2)
		self.preButR = Button(buttonPanel, text='Pb', width=2, command=self.preButFilCmd, bg="#1982C4").grid(row=2,column=3)
		self.preLab = Label(buttonPanel, text="Update Preview Photo", bg="#8ac926").grid(sticky='E', row=2,column=0, columnspan=2)
		
		self.calButL = Button(buttonPanel, text='Cc', width=2, command=self.calButCmd, bg="#1982C4").grid(row=3,column=2)
		self.calButR = Button(buttonPanel, text='Cb', width=2, command=self.calButFilCmd, bg="#1982C4").grid(row=3,column=3)
		self.calLab = Label(buttonPanel, text="Take Calibration Photo", bg="#8ac926").grid(sticky='E', row=3,column=0,columnspan=2)
		
		self.samBut = Button(buttonPanel, text='Sc', width=2, command=self.samButCmd, bg="#1982C4").grid(row=4,column=2)
		self.samBut = Button(buttonPanel, text='Sb', width=2, command=self.samButFilCmd, bg="#1982C4").grid(row=4,column=3)
		self.samLab = Label(buttonPanel, text="Take Seed Sample Photo", bg="#8ac926").grid(sticky='E', row=4,column=0,columnspan=2)
		
		self.graBut = Button(buttonPanel, text='Grade', width=4, command=self.graButCmd, bg="#1982C4").grid(row=5,column=2)
		self.graButSav = Button(buttonPanel, text='Save', width=4, command=self.graButSavCmd, bg="#1982C4").grid(row=5,column=3)
		self.graLab = Label(buttonPanel, text="Grade Seed Sample", bg="#8ac926").grid(sticky='E', row=5,column=0,columnspan=2)
		
		self.conBut = Button(buttonPanel, text='Tl', width=2, command=self.conButCmdL, bg="#1982C4").grid(sticky='E',row=6,column=2)
		self.conBut = Button(buttonPanel, text='Tr', width=2, command=self.conButCmdR, bg="#1982C4").grid(sticky='W',row=6,column=3)		
		self.conLab = Label(buttonPanel, text="Toggle Confidence", bg="#8ac926").grid(sticky='E', row=6,column=0, columnspan=2)

		
		"""
		Editable Settings:
		calib x1		= x of left top corner for cropping
		calib y1		= y of left top corner for cropping
		calib x2		= x of right bot corner for cropping 
		calib y2		= y of right bot corner for cropping
		calib update	= update the calibration image overlay
		
		sample x1		= x of left top corner for cropping
		sample y1		= y of left top corner for cropping
		sample x2		= x of right bot corner for cropping 
		sample y2		= y of right bot corner for cropping
		
		sample rows		= number of rows in the sample 
		sample columns	= number of columns in the sample
		
		sample update 	= update the sample image overlay 

		Displaying information:
		classification 	= grade of the sample
		loading 		= current timing of the sample grading
		log				= text box of all information
		"""
		#Calibration Cropping coordinates
		self.infCalCroX1StrVar = StringVar()
		self.infCalCroX1 = Entry(buttonPanel, textvariable=self.infCalCroX1StrVar, width=10).grid(sticky='W',row=8,column=1)
		self.infCalCroX1Lab = Label(buttonPanel, text="x1", bg="#8ac926").grid(sticky='E', row=8,column=0)

		self.infCalCroY1StrVar = StringVar()
		self.infCalCroY1 = Entry(buttonPanel, textvariable=self.infCalCroY1StrVar, width=10).grid(sticky='W',row=8,column=3)
		self.infCalCroY1Lab = Label(buttonPanel, text="y1", bg="#8ac926").grid(sticky='E', row=8,column=2)
		
		self.infCalCroX2StrVar = StringVar()
		self.infCalCroX2 = Entry(buttonPanel, textvariable=self.infCalCroX2StrVar, width=10).grid(sticky='W',row=9,column=1)
		self.infCalCroX2Lab = Label(buttonPanel, text="x2", bg="#8ac926").grid(sticky='E', row=9,column=0)
		
		self.infCalCroY2StrVar = StringVar()
		self.infCalCroY2 = Entry(buttonPanel, textvariable=self.infCalCroY2StrVar, width=10).grid(sticky='W',row=9,column=3)
		self.infCalCroY2Lab = Label(buttonPanel, text="y2", bg="#8ac926").grid(sticky='E', row=9,column=2)

		self.infCalCroDat = ((0.,0.),(1.,1.))
		self.infCalCroLab = Button(buttonPanel, text="Edit Calibration Crop", command=self.infCalCroCmd, bg="#6A4C93").grid(row=7,column=0,columnspan=4)
		
		#Sample Cropping coordinates
		self.infSamCroX1StrVar = StringVar()
		self.infSamCroX1 = Entry(buttonPanel, textvariable=self.infSamCroX1StrVar, width=10).grid(sticky='W',row=11,column=1)
		self.infSamCroX1Lab = Label(buttonPanel, text="x1", bg="#8ac926").grid(sticky='E', row=11,column=0)

		self.infSamCroY1StrVar = StringVar()
		self.infSamCroY1 = Entry(buttonPanel, textvariable=self.infSamCroY1StrVar, width=10).grid(sticky='W',row=11,column=3)
		self.infSamCroY1Lab = Label(buttonPanel, text="y1", bg="#8ac926").grid(sticky='E', row=11,column=2)
		
		self.infSamCroX2StrVar = StringVar()
		self.infSamCroX2 = Entry(buttonPanel, textvariable=self.infSamCroX2StrVar, width=10).grid(sticky='W',row=12,column=1)
		self.infSamCroX2Lab = Label(buttonPanel, text="x2", bg="#8ac926").grid(sticky='E', row=12,column=0)
		
		self.infSamCroY2StrVar = StringVar()
		self.infSamCroY2 = Entry(buttonPanel, textvariable=self.infSamCroY2StrVar, width=10).grid(sticky='W',row=12,column=3)
		self.infSamCroY2Lab = Label(buttonPanel, text="y2", bg="#8ac926").grid(sticky='E', row=12,column=2)
		
		self.infSamCroDat = ((0.,0.),(1.,1.))
		self.infSamCroLab = Button(buttonPanel, text="Edit Sample Crop", command=self.infSamCroCmd, bg="#6A4C93").grid(row=10,column=0,columnspan=4)

		#Sample Grid parameters
		self.infSamGriRowStrVar = StringVar()
		self.infSamGriRow = Entry(buttonPanel, textvariable=self.infSamGriRowStrVar, width=10).grid(sticky='W',row=14,column=1)
		self.infSamGriRowLab = Label(buttonPanel, text="Rows", bg="#8ac926").grid(sticky='E', row=14,column=0)

		self.infSamGriColStrVar = StringVar()
		self.infSamGriCol = Entry(buttonPanel, textvariable=self.infSamGriColStrVar, width=10).grid(sticky='W', row=14,column=3)
		self.infSamGriColLab = Label(buttonPanel, text="Columns", bg="#8ac926").grid(sticky='E', row=14,column=2)
		
		self.infSamGriDat = (1,1)
		self.infSamGriLab = Button(buttonPanel, text="Edit Sample Grid", command=self.infSamGriCmd, bg="#6A4C93").grid(row=13,column=0,columnspan=4)
		
		#The information canvas panel will be below the button panel
		self.infClaStrVar = StringVar()
		self.infClaStrVar.set("No grade yet.")
		self.infClaLabVar = Label(buttonPanel, textvariable=self.infClaStrVar, bg="#FFCA3A").grid(sticky='W',row=19,column=2, columnspan=2)
		self.infClaLab = Label(buttonPanel, text="Grade of Sample:", bg="#8ac926").grid(sticky='E',row=19,column=0, columnspan=2)
		
		self.infLogStrVar = StringVar()
		self.infLogStrVar.set("No grading completed.")
		self.infLogLabVar = Label(buttonPanel, textvariable=self.infLogStrVar, bg="#FFCA3A").grid(sticky='W',row=20,column=2, columnspan=2)
		self.infLogLab = Label(buttonPanel, text="Grading Timing:", bg="#8ac926").grid(sticky='E',row=20,column=0, columnspan=2)
		
		#advanced settings button
		self.advSetCount = 0
		self.advSetBut = Button(buttonPanel, text="Advanced Settings", command=self.advSetButCmd, bg="#fe9f0f").grid(row=18,column=0,columnspan=4)
		self.advSetWindow = None
		
		#Text log
		self.infLogTxt = Text(buttonPanel, bg="Grey", width=40, height=16, wrap=WORD)
		self.infLogTxt.grid(row=21,column=0, columnspan=4)
		self.infLogTxt.insert(END, "Welcome!\nFor reference, Red=DGR, Yellow=Fine, and Blue=Absent.")
		self.infLogTxt.insert(END, "\nAlso, The lighter the green, the higher the confidence.")
		self.infLogTxt.insert(END, "\nAlso, use the keys 'q', 'w', 'e', 'r', and 't' to toggle analysis images once a grading is complete.")		
		
		self.infLogScr = Scrollbar(buttonPanel, command=self.infLogTxt.yview)
		self.infLogScr.grid(row=21, column=4, sticky='nsew')
		self.infLogTxt['yscrollcommand'] = self.infLogScr.set
		
		"""
		4 Image Labels for display 
		preDisplayCal 	= Display of camera preview, with calibration crop overlay 
		preDisplayGra 	= Display of camera preview, with seed sample grid overlay
		calDisplay 		= Display of averaged calibration thumbnail image
		graDisplay 		= Display of graded seed sample image, with other confidence images as toggled.
		"""
		#Create image canvas panel
		imagePanel = Canvas(master, width=800, height=hMin, bg="#71A520")
		imagePanel.pack(side=TOP)
		
		#Set the images in the labels
		self.preDisCalImgDim = (250,200)
		self.rawCalImg = blankImg(self.preDisCalImgDim)
		self.calibUpdated = False
		self.preDisCalImg = convImg(iForm.drawBox(blankImg(self.preDisCalImgDim),(0,0),self.preDisCalImgDim))
		self.preDisCal = Label(imagePanel, image=self.preDisCalImg, width=self.preDisCalImgDim[0], height=self.preDisCalImgDim[1])
		self.preDisCal.grid(row=1,column=0, padx=5, pady=5)
		self.preDisCalLab = Label(imagePanel, text="Preview Image for Calibration", bg="#71A520").grid(row=0,column=0, pady=2)
		
		self.preDisGraImgDim = (250,200)
		self.rawSamImg = blankImg(self.preDisGraImgDim)
		self.preDisGraImg = convImg(iForm.drawGrid(blankImg(self.preDisGraImgDim),(0,0),self.preDisGraImgDim,1,1))
		self.preDisGra = Label(imagePanel, image=self.preDisGraImg, width=self.preDisGraImgDim[0], height=self.preDisGraImgDim[1])
		self.preDisGra.grid(row=1,column=1, padx=5, pady=5)
		self.preDisGraLab = Label(imagePanel, text="Preview Image for Seed Sample", bg="#71A520").grid(row=0, column=1)
		
		self.calDisImgDim = (250,200)
		self.calDisImg = convImg(blankImg(self.calDisImgDim))
		self.useCalImg = blankImg(self.calDisImgDim)
		self.calDis = Label(imagePanel, image=self.calDisImg, width=self.calDisImgDim[0], height=self.calDisImgDim[1])
		self.calDis.grid(sticky="NW", row=1,column=2, padx=5, pady=5)
		self.calDisLab = Label(imagePanel, text= "Calibration Image", bg="#71A520").grid(row=0,column=2)
		
		self.graDisImgDim = (500,400)
		self.graDisImg = convImg(iForm.drawGrid(blankImg(self.graDisImgDim),(0,0),self.graDisImgDim,1,1))
		self.sampleUpdated = False
		self.graDis = Label(imagePanel, image=self.graDisImg, width=self.graDisImgDim[0], height=self.graDisImgDim[1])
		self.graDis.grid(row=3,column=0, rowspan=2, columnspan=2, padx=5, pady=5)
		self.graDisLab = Label(imagePanel, text= "Seed Sample Image:", bg="#71A520").grid(sticky='E',row=2, column=0)
		self.graDisLabStrVar = StringVar()
		self.graDisLabStrVar.set("No Image Yet.")
		
		self.graDisCount = 0
		self.graDisExpWin = None
		self.graDisLabBut = Button(imagePanel, textvariable=self.graDisLabStrVar, command=self.graDisLabCmd, bg="#FFCA3A").grid(sticky='W',row=2,column=1)
		
		self.exaDisImgDim = (250,200)
		self.exaDisImg = convImg(blankImg(self.exaDisImgDim))
		self.exaDis = Label(imagePanel, image=self.exaDisImg, width=self.exaDisImgDim[0], height=self.exaDisImgDim[1])
		self.exaDis.grid(sticky="NW", row=3,column=2, padx=5, pady=5)
		self.exaDisLab = Label(imagePanel, text= "Examined Seed", bg="#71A520").grid(row=2, column=2)
		
		self.exaAnaTxt = Text(imagePanel, bg="#FFCA3A", width=30, height=11)
		self.exaAnaTxt.grid(sticky="NW", row=4, column=2, padx=5, pady=5)
		self.exaAnaTxt.insert(END,"No Grade Yet.")
		self.exaAnaTxt.config(state=DISABLED)
		
		
		"""
		Manage initial toggle img states.
		Add key bindings for the toggle img state.
		"""
		self.toggleState=0
		self.togImgWm = convImg(iForm.drawGrid(blankImg(self.graDisImgDim),(0,0),self.graDisImgDim,1,1))
		self.togImgRel = convImg(iForm.drawGrid(blankImg(self.graDisImgDim),(0,0),self.graDisImgDim,1,1))
		self.togImgDGR = convImg(iForm.drawGrid(blankImg(self.graDisImgDim),(0,0),self.graDisImgDim,1,1))
		self.togImgConf = convImg(iForm.drawGrid(blankImg(self.graDisImgDim),(0,0),self.graDisImgDim,1,1))
		
		"""
		Manage click bindings for seed examination.
		"""
		master.bind("<Key>", self.setToggleStatus)
		self.graDis.bind("<Button-1>", self.getMousePressLeft)
		self.graDis.bind("<Button-3>", self.getMousePressRight)
		
		self.popup = Menu(master, tearoff=0)
		self.popup.add_command(label="Mark as DGR", command=self.setSeedCustDGR)
		self.popup.add_command(label="Mark as not DGR", command=self.setSeedCustNotDGR)
		self.popup.add_command(label="Mark as NaN", command=self.setSeedCustNaN)
		self.popup.add_separator()
		self.popup.add_command(label="Reset Seed Grade", command=self.resetSeedGrade)
	
	#Update a label with the given image
	def updateLabelImg(self, label, img):
		label.configure(image=img)
		label.image=img
	
	#Update the preview image displays using the given image
	#Apply grid and crop overlays here
	def updatePreviewImages(self, img):
		calibPrevImg = convImg(iForm.drawBox(iForm.resizeImg(img, self.preDisCalImgDim[0],
							   self.preDisCalImgDim[1]), self.infCalCroDat[0], self.infCalCroDat[1])) 
		seedSampleImg = convImg(iForm.drawGrid(iForm.resizeImg(img, self.preDisGraImgDim[0],self.preDisGraImgDim[1]), 
								self.infSamCroDat[0], self.infSamCroDat[1], self.infSamGriDat[0], self.infSamGriDat[1]))
		
		#Display calibration preview
		if not self.calibUpdated: 
			self.rawCalImg = img
		self.updateLabelImg(self.preDisCal, calibPrevImg)
		
		#Display seed sample preview
		if not self.sampleUpdated:
			self.rawSamImg = img
		self.updateLabelImg(self.preDisGra, seedSampleImg)
	
	#Update the calibration image display using the given image
	def updateCalibImage(self, ogImg, aveImg):
		self.calibUpdated = True
		
		calibrationImg = convImg(iForm.resizeImg(aveImg, self.calDisImgDim[0], self.calDisImgDim[1]))
		
		self.newCamImg = convImg(iForm.drawBox(iForm.resizeImg(ogImg, self.preDisCalImgDim[0],self.preDisCalImgDim[1]), 
								 self.infCalCroDat[0], self.infCalCroDat[1]))
						 
		self.updateLabelImg(self.calDis, calibrationImg)
		
		#Update preview images
		#Display calibration preview
		self.updateLabelImg(self.preDisCal, self.newCamImg)
	
	#Update the seed sample image display using the given image
	def updateSampleImage(self, img):
		self.sampleUpdated= True
		self.rawSamImg = img
		self.updateLabelImg(self.graDis, convImg(iForm.drawGrid(iForm.resizeImg(iForm.cropImg(img, self.infSamCroDat[0], 
							self.infSamCroDat[1]), self.graDisImgDim[0], self.graDisImgDim[1]), (0.,0.),(1.,1.),self.infSamGriDat[0], self.infSamGriDat[1])))
	
		#Display seed sample preview
		self.updateLabelImg(self.preDisGra, convImg(iForm.drawGrid(iForm.resizeImg(img, self.preDisGraImgDim[0], 
							self.preDisGraImgDim[1]), self.infSamCroDat[0], self.infSamCroDat[1], self.infSamGriDat[0], self.infSamGriDat[1])))
	
	#Start using a phone camera through IP Webcam 
	def ipCamButCmd(self):
		succMsg = iCamr.switchToIPCam("http://" + str(self.ipCamIPStrVar.get()));
		self.infLogTxt.insert(END, "\n" + succMsg)
	
	#Take and display the preview image
	def preButCmd(self):
		#Get and convert photos
		print("preview button pressed.")
		self.newCamImg = convImg(iCamr.takePhoto())	
		
		#Update GUI
		self.updatePreviewImages(iCamr.getMostRecentPhoto())
		
	#Take and display the calibration image
	def calButCmd(self):
		#Set calibration image
		print("calibration button pressed")
		ogCalibrationImg, ogOGImg, self.calibFilePath = iCali.takeCalibrationPhoto(leftTopCorner=self.infCalCroDat[0], rightBotCorner=self.infCalCroDat[1])
		self.rawCalImg = ogOGImg
		self.useCalImg = ogCalibrationImg
		self.updateCalibImage(ogOGImg, ogCalibrationImg)

		
	#Take and display the seed sample image
	def samButCmd(self):
		#Set sample image
		print("seed sample button pressed")
		self.newCamImg = convImg(iCamr.takePhoto())
				
		#Save the image
		i = 0
		while os.path.exists(os.path.join("Images", "guiSeedSamplePhoto_{}.png".format(i))):
			i = i + 1
		self.sampleFilePath = os.path.join("Images", "guiSeedSamplePhoto_{}.png".format(i))
		iForm.saveImageToFolder("Images", "guiSeedSamplePhoto_{}.png".format(i), iCamr.getMostRecentPhoto())
		
		#Update the sample seed images
		self.updateSampleImage(iCamr.getMostRecentPhoto())
		
	#Browse for and display the preview image
	def preButFilCmd(self):
		print("preview browse button pressed.")
		tempPath = browseImageFromComp(0)
		if len(tempPath) > 0:
			self.preButFilPath = tempPath
			preImg = iForm.readImage(self.preButFilPath)
			self.updatePreviewImages(preImg)
		
	#Browse for and display the calibration image
	def calButFilCmd(self):
		print("calibration browse button pressed")
		self.preButFilPath = browseImageFromComp(0)
		self.calibFilePath = self.preButFilPath
		if len(self.preButFilPath) > 0:
			ogCalibrationImg, ogOGImg, temp = iCali.takeCalibrationPhoto(leftTopCorner=self.infCalCroDat[0], 
										rightBotCorner=self.infCalCroDat[1], filePath=self.preButFilPath)	
			self.rawCalImg=ogOGImg
			self.useCalImg = ogCalibrationImg
			
			self.updateCalibImage(ogOGImg, ogCalibrationImg)
		
	#Browse for and display the seed sample image
	def samButFilCmd(self):
		print("sample browse button pressed")
		tempPath = browseImageFromComp(0)
		self.sampleFilePath = tempPath
		if len(tempPath) > 0:
			self.preButFilPath = tempPath
			sampleImg = iForm.readImage(self.preButFilPath)
			self.updateSampleImage(sampleImg)
		
	#Grade and display the graded seed sample
	def graButCmd(self):
		#Update status
		print("grading button pressed")
		
		#Stop if there is no calibration image.
		if np.array_equal(self.useCalImg, blankImg(self.calDisImgDim)) or self.sampleUpdated==False:
			return
		
		#Divide the whole image into the specified individual seed images
		self.seedSampleList = iForm.gridPartitionImg(iForm.cropImg(self.rawSamImg, self.infSamCroDat[0], self.infSamCroDat[1]), self.infSamGriDat[0], self.infSamGriDat[1])	
		
		#Classify each seed and associate the data with each seed
		self.seedSampleInfo, timeStr = sClas.classifySeedSample(self.seedSampleList, self.useCalImg)	
		self.seedSampleAreas = [seedInfo[2] for seedInfo in self.seedSampleInfo]
		self.seedSampleInfoMod = copy.deepcopy(self.seedSampleInfo)
		
		self.printGradeToggleImgs(self.seedSampleInfo, timeStr)
	
	#Save the current grading to a text report.
	def graButSavCmd(self):
		"""
		Example Report Format:
		~~~~~~~~~~~~~Canola Seed Grading~~~~~~~~~~~~~
		Date: 		Tue Aug 16 21:30:00 1988 (en_US)
		Grade:		No. 1
		
		Distinctly Green Sample %:	0.606%
		Distinctly Green Seeds:		3
		Used Seed Sample Size:		495
		Initial Seed Sample Size: 	500
		Discarded Seeds:			5
		Unaltered Grade:			Yes
		
		Calibration Image Used:		CalibrationImages\guiCalibrationPhoto.png
		Seed Sample Image Used:		Images\guiSeedSamplePhoto_0.png
		
		D = Distinctly Green, G = Not Distinctly Green, N = Discarded Seed
					Col
					1	2	3	4	...
		Row 	1	D	G	G 	G
				2	G	N	G	G
				3	G 	G 	G	G
				...
		"""
		#Parse variables
		aNums = re.findall(r'\d+', self.seedAnalysisString[0])
		dgrPercent = float(aNums[3])+float(aNums[4])/1000
		discardQuant = aNums[5] if len(aNums)>5 else 0
		gradeUnaltered = True
		for i in self.seedSampleInfoMod:
			if str(self.seedSampleInfoMod[1])[:5] =='cust_':
				gradeUnaltered = False 
				break
		cols=''
		for col in range(self.infSamGriDat[1]):
			cols = cols+'	'+str(col+1)
		
		rows=''
		for row in range(self.infSamGriDat[0]):
			rows = rows + str(row+1)
			for col in range(self.infSamGriDat[1]):
				rows = rows + '	' + self.dgrToString(self.seedSampleInfoMod[row*self.infSamGriDat[1]+col][1], char=True, asterisk=True)
			rows = rows + '\n	'
		
		report = """
~~~~~~~~~~~~~Canola Seed Grading~~~~~~~~~~~~~
Date: 		{}
Grade:		{}

Distinctly Green Sample %:	{:5.3f}%
Distinctly Green Seeds:		{}
Used Seed Sample Size:		{}
Initial Seed Sample Size: 	{}
Discarded Seeds:		{}
Unaltered Grade:		{}

Calibration Image Used:		{}
Seed Sample Image Used:		{}
Time to Grade:			{}

D = Distinctly Green, G = Not Distinctly Green, N = Discarded Seed
		Col
	{}
Row 	{}

More Settings:
Seed Sample Crop X1:		{}
Seed Sample Crop Y1:		{}
Seed Sample Crop X2:		{}
Seed Sample Crop Y2:		{}

{}

		""".format(
			time.strftime("%c"), self.infClaStrVar.get(), 
			dgrPercent, aNums[1], aNums[2], int(aNums[2])+int(discardQuant), discardQuant, gradeUnaltered,
			self.calibFilePath, self.sampleFilePath, self.infLogStrVar.get(),
			cols, rows, 
			self.infSamCroDat[0][0], self.infSamCroDat[0][1], self.infSamCroDat[1][0], self.infSamCroDat[1][1],
			sClas.getASettingsString())

		#Get the current date
		reportDate = time.strftime("%Y_%b_%d_%H_%M_%S")
		
		#Get the content of the report
		
		#Get the file name of the report
		reportFilename = r"Canola_Grading_{}.txt".format(reportDate)
		
		#Save the report as a text file
		reportFile = open(os.path.join("Reports", reportFilename), 'w')
		reportFile.write(report)
		self.infLogTxt.insert(END,"\nReport saved.")
		
	#Print the grade results and set the toggle images
	def printGradeToggleImgs(self, seedSampleInfo, timeStr=None, toggleState=0):
		#Analyze the total data of the seed sample
		seedAnalysisString = sClas.getSeedAnly().analyzeSeedSample(seedSampleInfo, timeStr)
		self.infLogTxt.insert(END, "\n"+seedAnalysisString[0])
		self.infClaStrVar.set(seedAnalysisString[0][36:42])
		if len(seedAnalysisString) > 1:
			self.infLogStrVar.set(seedAnalysisString[1])
		self.seedAnalysisString = seedAnalysisString
		
		#Create images for toggling
		self.togImgWm, self.togImgRel, self.togImgDGR, self.togImgConf = sClas.getSeedAnly().createInfoImages(seedSampleInfo, self.infSamGriDat[0], self.infSamGriDat[1])
		self.graDisLabStrVar.set("Raw Image")
		self.toggleState=toggleState
		self.updateToggleImg()
	
	#Toggle the graded seed sample visual information (Left)
	def conButCmdL(self):		
		print("left toggle button pressed")
		
		#Check if a seed sample has been graded yet.
		if self.infClaStrVar.get() == "No grade yet.":
			self.infLogTxt.insert(END, "\nNo seed sample graded yet.")
			return
		
		#If the sample has been graded, toggle state/picture
		self.toggleState = (self.toggleState-1)%5
		
		#Update the analysis state/picture
		self.updateToggleImg()
	
	#Toggle the graded seed sample visual information (Right)
	def conButCmdR(self):
		print("right toggle button pressed")
		
		#Check if a seed sample has been graded yet.
		if self.infClaStrVar.get() == "No grade yet.":
			self.infLogTxt.insert(END, "\nNo seed sample graded yet.")
			return
		
		#If the sample has been graded, toggle state/picture
		self.toggleState = (self.toggleState+1)%5
		
		#Update the analysis state/picture
		self.updateToggleImg()
	
	#Set the graded seed sample visual information based on the user key press
	def setToggleStatus(self, event):
		#print("Keyboard pressed: ", repr(event.char))
			
		#Check if a seed sample has been graded yet.
		if self.infClaStrVar == "No grade yet.":
			self.infLogTxt.insert(END, "No seed sample graded yet.")
			return
		
		keyPress = str(repr(event.char))[1:2]
		keyMapping = {'q':0,'w':1,'e':2,'r':3,'t':4}
		
		if keyPress in keyMapping.keys():
			self.toggleState = keyMapping[keyPress]
			
			self.updateToggleImg()
		
	#Update the analysis state/picture
	def updateToggleImg(self):
		if self.toggleState==0:
			self.updateSampleImage(self.rawSamImg)
			self.graDisLabStrVar.set("Raw Image")

		elif self.toggleState==1:
			self.updateLabelImg(self.graDis, convImg(iForm.resizeImg(self.togImgWm, self.graDisImgDim[0], self.graDisImgDim[1])))
			self.graDisLabStrVar.set("Graded Seeds Image")
			
		elif self.toggleState==2:
			self.updateLabelImg(self.graDis, convImg(iForm.resizeImg(self.togImgRel, self.graDisImgDim[0], self.graDisImgDim[1])))
			self.graDisLabStrVar.set("Graded Smears Image")
			
		elif self.toggleState==3:
			self.updateLabelImg(self.graDis, convImg(iForm.resizeImg(self.togImgDGR, self.graDisImgDim[0], self.graDisImgDim[1])))
			self.graDisLabStrVar.set("Graded DGR Pixels Image")
			
		elif self.toggleState==4:
			self.updateLabelImg(self.graDis, convImg(iForm.resizeImg(self.togImgConf, self.graDisImgDim[0], self.graDisImgDim[1])))
			self.graDisLabStrVar.set("Grading Confidence Image")
			
	
	#Update the preview calibration crop settings and image
	def infCalCroCmd(self):
		
		#Check the validity of crop parameters
		errorMsg = "\nError: Calibration Crop Parameters must be numbers in the range of [0.0,1.0]."
		
		getCoordinateLow = lambda x: float(x) if len(x)>0 else 0.
		getCoordinateHigh = lambda x: float(x) if len(x)>0 else 1.
		try:
			newX1 = getCoordinateLow(self.infCalCroX1StrVar.get())
			newY1 = getCoordinateLow(self.infCalCroY1StrVar.get())
			newX2 = getCoordinateHigh(self.infCalCroX2StrVar.get())
			newY2 = getCoordinateHigh(self.infCalCroY2StrVar.get())
			if newX1<0. or newX1>1. or newY1<0. or newY1>1. or newX2<0. or newX2>1. or newY2<0. or newY2>1.:
				self.infLogTxt.insert(END, errorMsg)
				return
		except ValueError:
			self.infLogTxt.insert(END, errorMsg)
			return
		
		#If the parameters are valid, update them and the images 
		self.infCalCroDat = ((newX1,newY1),(newX2,newY2))
				
		ogCalibrationImg, ogOGImg, self.calibFilePath = iCali.takeCalibrationPhoto(img=self.rawCalImg,leftTopCorner=self.infCalCroDat[0], 
															   rightBotCorner=self.infCalCroDat[1])
		self.useCalImg = ogCalibrationImg
		self.updateCalibImage(ogOGImg, ogCalibrationImg)
		

		
	#Update the preview and full sample crop settings and image
	def infSamCroCmd(self):
		#Check the validity of crop parameters
		errorMsg = "\nError: Seed Sample Crop Parameters must be numbers in the range of [0.0,1.0]."
		getCoordinateLow = lambda x: float(x) if len(x)>0 else 0.
		getCoordinateHigh = lambda x: float(x) if len(x)>0 else 1.
		try:
			newX1 = getCoordinateLow(self.infSamCroX1StrVar.get())
			newY1 = getCoordinateLow(self.infSamCroY1StrVar.get())
			newX2 = getCoordinateHigh(self.infSamCroX2StrVar.get())
			newY2 = getCoordinateHigh(self.infSamCroY2StrVar.get())
			if newX1<0. or newX1>1. or newY1<0. or newY1>1. or newX2<0. or newX2>1. or newY2<0. or newY2>1. or newX1 >= newX2 or newY1 >= newY2:
				self.infLogTxt.insert(END, errorMsg)
				return
		except ValueError:
			self.infLogTxt.insert(END, errorMsg)
			return 
		
		#If the parameters are valid, update them and the images 
		self.infSamCroDat = ((newX1,newY1),(newX2,newY2))
		
		self.updateSampleImage(self.rawSamImg)
		
	#Update the preview and full sample crop settings and image
	def infSamGriCmd(self):
		#Check the validity of crop parameters
		errorMsg = "\nError: Seed Sample Grid Parameters must be numbers greater than or equal to 1."
		getGrid = lambda x: int(x) if len(x)>0 else 1
		try:
			newRow = getGrid(self.infSamGriRowStrVar.get())
			newCol = getGrid(self.infSamGriColStrVar.get())

			if newRow < 1 or newCol < 1:
				self.infLogTxt.insert(END, errorMsg)
				return
		except ValueError:
			self.infLogTxt.insert(END, errorMsg)
			return 
		
		#If the parameters are valid, update them and the images 
		self.infSamGriDat = (newRow,newCol)
		self.updateSampleImage(self.rawSamImg)
	
	#Enlarges the current grading image in a new window if there is one present.
	def graDisLabCmd(self):
		#ensure only one settings window is open at any time
		if self.infClaStrVar.get() == "No grade yet.":
			self.infLogTxt.insert(END, "\nNo seed sample graded yet.")
			return
		if self.graDisCount > 0:
			return
		self.graDisCount = self.graDisCount + 1
		
		#Create the window
		self.graDisExpWin = Toplevel(self)
		self.graDisExpWin.wm_title(self.graDisLabStrVar.get())
		self.graDisExpWin.protocol("WM_DELETE_WINDOW", self.closeSampleImgWindow)
		
		#Add the image to the window
		self.graDisExpWin.mainImgDim = (880,660)
		self.graDisExpWin.mainImg = Label(self.graDisExpWin, image=convImg(blankImg((100, 100))))
		self.graDisExpWin.mainImg.pack()
		
		if self.toggleState==0:												
			self.updateLabelImg(self.graDisExpWin.mainImg, convImg(iForm.drawGrid(iForm.resizeImg(iForm.cropImg(self.rawSamImg, self.infSamCroDat[0], 
								self.infSamCroDat[1]), self.graDisExpWin.mainImgDim[0], self.graDisExpWin.mainImgDim[1]), (0.,0.),(1.,1.),self.infSamGriDat[0], self.infSamGriDat[1])))

		elif self.toggleState==1:
			self.updateLabelImg(self.graDisExpWin.mainImg, convImg(iForm.resizeImg(self.togImgWm, self.graDisExpWin.mainImgDim[0], self.graDisExpWin.mainImgDim[1])))
			
		elif self.toggleState==2:
			self.updateLabelImg(self.graDisExpWin.mainImg, convImg(iForm.resizeImg(self.togImgRel, self.graDisExpWin.mainImgDim[0], self.graDisExpWin.mainImgDim[1])))
			
		elif self.toggleState==3:
			self.updateLabelImg(self.graDisExpWin.mainImg, convImg(iForm.resizeImg(self.togImgDGR, self.graDisExpWin.mainImgDim[0], self.graDisExpWin.mainImgDim[1])))
			
		elif self.toggleState==4:
			self.updateLabelImg(self.graDisExpWin.mainImg, convImg(iForm.resizeImg(self.togImgConf, self.graDisExpWin.mainImgDim[0], self.graDisExpWin.mainImgDim[1])))

		#TODO
	
	
	#close the expanded graded image window
	def closeSampleImgWindow(self):
		self.graDisCount = self.graDisCount - 1
		self.graDisExpWin.destroy()
	
	#Opens an advanced settings window
	def advSetButCmd(self):
		#ensure only one settings window is open at any time
		if self.advSetCount > 0:
			return
		self.advSetCount = self.advSetCount + 1
		
		#Create the window 
		self.advSetWindow = Toplevel(self)
		self.advSetWindow.wm_title("Advanced Settings")
		self.advSetWindow.protocol("WM_DELETE_WINDOW", self.closeAdvSetWindow)
		
		#Set buttons and entries
		#Background
		self.advSetWindow.bgTtl = Label(self.advSetWindow, text="Background Parse Settings", bg='#6b7560').grid(row=0, column=0, columnspan=4)
		
		self.advSetWindow.bgSatLab = Label(self.advSetWindow, text="General Saturation Max:", bg='#8e9c80').grid(sticky='E',row=1, column=0, columnspan=2)
		self.advSetWindow.bgSatLim = Label(self.advSetWindow, text=str(sClas.getASettings('bgdSatMax', value='limits')), bg='#8e9c80').grid(row=1, column=2)
		self.advSetWindow.bgSatStrVar = StringVar()
		self.advSetWindow.bgSatStrVar.set(sClas.getASettings('bgdSatMax', value='actual'))
		self.advSetWindow.bgSatEnt = Entry(self.advSetWindow, textvariable=self.advSetWindow.bgSatStrVar, width=10).grid(row=1, column=3)
		
		self.advSetWindow.bgGSatLab = Label(self.advSetWindow, text="Green Saturation Max:", bg='#8e9c80').grid(sticky='E',row=2, column=0, columnspan=2)
		self.advSetWindow.bgGSatLim = Label(self.advSetWindow, text=str(sClas.getASettings('bgdNGSatMax', value='limits')), bg='#8e9c80').grid(row=2, column=2)
		self.advSetWindow.bgGSatStrVar = StringVar()
		self.advSetWindow.bgGSatStrVar.set(sClas.getASettings('bgdNGSatMax', value='actual'))
		self.advSetWindow.bgGSatEnt = Entry(self.advSetWindow, textvariable=self.advSetWindow.bgGSatStrVar, width=10).grid(row=2, column=3)

		self.advSetWindow.bgGBriLab = Label(self.advSetWindow, text="Green Brightness Max:", bg='#8e9c80').grid(sticky='E', row=3, column=0, columnspan=2)
		self.advSetWindow.bgGBriLim = Label(self.advSetWindow, text=str(sClas.getASettings('bgdNGBriMin', value='limits')), bg='#8e9c80').grid(row=3, column=2)
		self.advSetWindow.bgGBriStrVar = StringVar()
		self.advSetWindow.bgGBriStrVar.set(sClas.getASettings('bgdNGBriMin', value='actual'))
		self.advSetWindow.bgGBriEnt = Entry(self.advSetWindow, textvariable=self.advSetWindow.bgGBriStrVar, width=10).grid(row=3, column=3)
		
		self.advSetWindow.newline = Label(self.advSetWindow, text="").grid(row=4)
		
		#Smear
		self.advSetWindow.smrTtl = Label(self.advSetWindow, text="Smear Parse Settings", bg='#7f9912').grid(row=5, column=0, columnspan=4)
		
		self.advSetWindow.smrThreLab = Label(self.advSetWindow, text="Threshold Min:", bg='#b5b516').grid(sticky='E',row=6, column=0, columnspan=2)
		self.advSetWindow.smrThreLim = Label(self.advSetWindow, text=str(sClas.getASettings('smrThreshMin', value='limits')), bg='#b5b516').grid(row=6, column=2)
		self.advSetWindow.smrThreStrVar = StringVar()
		self.advSetWindow.smrThreStrVar.set(sClas.getASettings('smrThreshMin', value='actual'))
		self.advSetWindow.smrThreEnt = Entry(self.advSetWindow, textvariable=self.advSetWindow.smrThreStrVar, width=10).grid(row=6, column=3)
		
		self.advSetWindow.smrHDifLab = Label(self.advSetWindow, text="Hue Difference Weight:", bg='#b5b516').grid(sticky='E',row=7, column=0, columnspan=2)
		self.advSetWindow.smrHDifLim = Label(self.advSetWindow, text=str(sClas.getASettings('smrDiffW', value='limits')), bg='#b5b516').grid(row=7, column=2)
		self.advSetWindow.smrHDifStrVar = StringVar()
		self.advSetWindow.smrHDifStrVar.set(sClas.getASettings('smrDiffW', value='actual'))
		self.advSetWindow.smrHDifEnt = Entry(self.advSetWindow, textvariable=self.advSetWindow.smrHDifStrVar, width=10).grid(row=7, column=3)

		self.advSetWindow.smrGPenLab = Label(self.advSetWindow, text="Grey Penalty Weight:", bg='#b5b516').grid(sticky='E', row=8, column=0, columnspan=2)
		self.advSetWindow.smrGPenLim = Label(self.advSetWindow, text=str(sClas.getASettings('smrGreyPenaltyW', value='limits')), bg='#b5b516').grid(row=8, column=2)
		self.advSetWindow.smrGPenStrVar = StringVar()
		self.advSetWindow.smrGPenStrVar.set(sClas.getASettings('smrGreyPenaltyW', value='actual'))
		self.advSetWindow.smrGPenEnt = Entry(self.advSetWindow, textvariable=self.advSetWindow.smrGPenStrVar, width=10).grid(row=8, column=3)

		self.advSetWindow.smrPxlLab = Label(self.advSetWindow, text="Smear Pixel Count Min:", bg='#b5b516').grid(sticky='E', row=9, column=0, columnspan=2)
		self.advSetWindow.smrPxlLim = Label(self.advSetWindow, text=str(sClas.getASettings('smrPxlMin', value='limits')), bg='#b5b516').grid(row=9, column=2)
		self.advSetWindow.smrPxlStrVar = StringVar()
		self.advSetWindow.smrPxlStrVar.set(sClas.getASettings('smrPxlMin', value='actual'))
		self.advSetWindow.smrPxlEnt = Entry(self.advSetWindow, textvariable=self.advSetWindow.smrPxlStrVar, width=10).grid(row=9, column=3)

		self.advSetWindow.newline = Label(self.advSetWindow, text="").grid(row=10)

		#DGR
		self.advSetWindow.dgrTtl = Label(self.advSetWindow, text="Distinctly Green Pixel Settings", bg='#2cb00e').grid(row=11, column=0, columnspan=4)
		
		self.advSetWindow.dgrThreLab = Label(self.advSetWindow, text="Threshold Min:", bg='#3be615').grid(sticky='E',row=12, column=0, columnspan=2)
		self.advSetWindow.dgrThreLim = Label(self.advSetWindow, text=str(sClas.getASettings('dgrThreshMin', value='limits')), bg='#3be615').grid(row=12, column=2)
		self.advSetWindow.dgrThreStrVar = StringVar()
		self.advSetWindow.dgrThreStrVar.set(sClas.getASettings('dgrThreshMin', value='actual'))
		self.advSetWindow.dgrThreEnt = Entry(self.advSetWindow, textvariable=self.advSetWindow.dgrThreStrVar, width=10).grid(row=12, column=3)
		
		self.advSetWindow.dgrRelLLab = Label(self.advSetWindow, text="Relative L Max:", bg='#3be615').grid(sticky='E',row=13, column=0, columnspan=2)
		self.advSetWindow.dgrRelLLim = Label(self.advSetWindow, text=str(sClas.getASettings('dgrLRelMax', value='limits')), bg='#3be615').grid(row=13, column=2)
		self.advSetWindow.dgrRelLStrVar = StringVar()
		self.advSetWindow.dgrRelLStrVar.set(sClas.getASettings('dgrLRelMax', value='actual'))
		self.advSetWindow.dgrRelLEnt = Entry(self.advSetWindow, textvariable=self.advSetWindow.dgrRelLStrVar, width=10).grid(row=13, column=3)
		
		self.advSetWindow.dgrLMinLab = Label(self.advSetWindow, text="L Min:", bg='#3be615').grid(sticky='E',row=14, column=0, columnspan=2)
		self.advSetWindow.dgrLMinLim = Label(self.advSetWindow, text=str(sClas.getASettings('dgrLMin', value='limits')), bg='#3be615').grid(row=14, column=2)
		self.advSetWindow.dgrLMinStrVar = StringVar()
		self.advSetWindow.dgrLMinStrVar.set(sClas.getASettings('dgrLMin', value='actual'))
		self.advSetWindow.dgrLMinEnt = Entry(self.advSetWindow, textvariable=self.advSetWindow.dgrLMinStrVar, width=10).grid(row=14, column=3)

		self.advSetWindow.dgrLDifLab = Label(self.advSetWindow, text="ab Volume Distance Max:", bg='#3be615').grid(sticky='E', row=15, column=0, columnspan=2)
		self.advSetWindow.dgrLDifLim = Label(self.advSetWindow, text=str(sClas.getASettings('dgrDistMax', value='limits')), bg='#3be615').grid(row=15, column=2)
		self.advSetWindow.dgrLDifStrVar = StringVar()
		self.advSetWindow.dgrLDifStrVar.set(sClas.getASettings('dgrDistMax', value='actual'))
		self.advSetWindow.dgrLDifEnt = Entry(self.advSetWindow, textvariable=self.advSetWindow.dgrLDifStrVar, width=10).grid(row=15, column=3)

		self.advSetWindow.dgrAEdgLab = Label(self.advSetWindow, text="a Edge Slope:", bg='#3be615').grid(sticky='E', row=16, column=0, columnspan=2)
		self.advSetWindow.dgrAEdgLim = Label(self.advSetWindow, text=str(sClas.getASettings('dgrAEdge', value='limits')), bg='#3be615').grid(row=16, column=2)
		self.advSetWindow.dgrAEdgStrVar = StringVar()
		self.advSetWindow.dgrAEdgStrVar.set(sClas.getASettings('dgrAEdge', value='actual'))
		self.advSetWindow.dgrAEdgEnt = Entry(self.advSetWindow, textvariable=self.advSetWindow.dgrAEdgStrVar, width=10).grid(row=16, column=3)
		
		self.advSetWindow.dgrBEdgLab = Label(self.advSetWindow, text="b Edge Slope:", bg='#3be615').grid(sticky='E', row=17, column=0, columnspan=2)
		self.advSetWindow.dgrBEdgLim = Label(self.advSetWindow, text=str(sClas.getASettings('dgrBEdge', value='limits')), bg='#3be615').grid(row=17, column=2)
		self.advSetWindow.dgrBEdgStrVar = StringVar()
		self.advSetWindow.dgrBEdgStrVar.set(sClas.getASettings('dgrBEdge', value='actual'))
		self.advSetWindow.dgrBEdgEnt = Entry(self.advSetWindow, textvariable=self.advSetWindow.dgrBEdgStrVar, width=10).grid(row=17, column=3)		

		self.advSetWindow.dgrFracLab = Label(self.advSetWindow, text="DG Fraction Min:", bg='#3be615').grid(sticky='E', row=18, column=0, columnspan=2)
		self.advSetWindow.dgrFracLim = Label(self.advSetWindow, text=str(sClas.getASettings('dgrFracMin', value='limits')), bg='#3be615').grid(row=18, column=2)
		self.advSetWindow.dgrFracStrVar = StringVar()
		self.advSetWindow.dgrFracStrVar.set(sClas.getASettings('dgrFracMin', value='actual'))
		self.advSetWindow.dgrFracEnt = Entry(self.advSetWindow, textvariable=self.advSetWindow.dgrFracStrVar, width=10).grid(row=18, column=3)	
	
		self.advSetWindow.newline = Label(self.advSetWindow, text="").grid(row=17)
		self.advSetWindow.saveSetBut = Button(self.advSetWindow, text='Save Settings', command=self.saveAdvSettings, bg="#c41019").grid(row=19, column=0, columnspan=4)
		self.advSetWindow.saveSetBut = Button(self.advSetWindow, text='Reset Settings', command=self.resetAdvSettings, bg="#c41019").grid(row=20, column=0, columnspan=4)		
		self.advSetWindow.statusMsgStrVar = StringVar()
		self.advSetWindow.statusMsg = Label(self.advSetWindow, textvariable=self.advSetWindow.statusMsgStrVar, width=40).grid(row=21, column=0, columnspan=4)

		
	#close the advanced settings window
	def closeAdvSetWindow(self):
		self.advSetCount = self.advSetCount - 1
		self.advSetWindow.destroy()
	
	#save the input settings of the advanced settings window
	#also saves the input savings to a txt file
	def saveAdvSettings(self):
		#Get the settings from the user
		settingsDict={}
		settingsDict['bgdSatMax']=float(self.advSetWindow.bgSatStrVar.get())		
		settingsDict['bgdNGSatMax']=float(self.advSetWindow.bgGSatStrVar.get())
		settingsDict['bgdNGBriMin']=float(self.advSetWindow.bgGBriStrVar.get())
		
		settingsDict['smrThreshMin']=float(self.advSetWindow.smrThreStrVar.get())
		settingsDict['smrDiffW']=float(self.advSetWindow.smrHDifStrVar.get())
		settingsDict['smrGreyPenaltyW']=float(self.advSetWindow.smrGPenStrVar.get())
		settingsDict['smrPxlMin']=float(self.advSetWindow.smrPxlStrVar.get())
		
		settingsDict['dgrThreshMin']=float(self.advSetWindow.dgrThreStrVar.get())
		settingsDict['dgrLRelMax']=float(self.advSetWindow.dgrRelLStrVar.get())
		settingsDict['dgrLMin']=float(self.advSetWindow.dgrLMinStrVar.get())
		settingsDict['dgrDistMax']=float(self.advSetWindow.dgrLDifStrVar.get())
		settingsDict['dgrAEdge']=float(self.advSetWindow.dgrAEdgStrVar.get())
		settingsDict['dgrBEdge']=float(self.advSetWindow.dgrBEdgStrVar.get())
		settingsDict['dgrFracMin']=float(self.advSetWindow.dgrFracStrVar.get())

		if(sClas.setASettings(settingsDict)):
			self.advSetWindow.statusMsgStrVar.set("Settings Successfully Saved.")
		else:
			self.advSetWindow.statusMsgStrVar.set("Range Error: Settings Unsuccessfully Saved.")

	#Reset the advanced settings to the defaults
	def resetAdvSettings(self):
		sClas.resetASettings()
		self.advSetWindow.bgSatStrVar.set(sClas.getASettings('bgdSatMax', value='actual'))
		self.advSetWindow.bgGSatStrVar.set(sClas.getASettings('bgdNGSatMax', value='actual'))
		self.advSetWindow.bgGBriStrVar.set(sClas.getASettings('bgdNGBriMin', value='actual'))

		self.advSetWindow.smrThreStrVar.set(sClas.getASettings('smrThreshMin', value='actual'))
		self.advSetWindow.smrHDifStrVar.set(sClas.getASettings('smrDiffW', value='actual'))
		self.advSetWindow.smrGPenStrVar.set(sClas.getASettings('smrGreyPenaltyW', value='actual'))
		self.advSetWindow.smrPxlStrVar.set(sClas.getASettings('smrPxlMin', value='actual'))
		
		self.advSetWindow.dgrThreStrVar.set(sClas.getASettings('dgrThreshMin', value='actual'))
		self.advSetWindow.dgrRelLStrVar.set(sClas.getASettings('dgrLRelMax', value='actual'))
		self.advSetWindow.dgrLMinStrVar.set(sClas.getASettings('dgrLMin', value='actual'))
		self.advSetWindow.dgrLDifStrVar.set(sClas.getASettings('dgrDistMax', value='actual'))
		self.advSetWindow.dgrAEdgStrVar.set(sClas.getASettings('dgrAEdge', value='actual'))
		self.advSetWindow.dgrBEdgStrVar.set(sClas.getASettings('dgrBEdge', value='actual'))
		self.advSetWindow.dgrFracStrVar.set(sClas.getASettings('dgrFracMin', value='actual'))
	
		self.advSetWindow.statusMsgStrVar.set("Settings Reset.")
	
	
	#Get the row and column of a mouse click on the sample analysis image
	def getClickRowAndCol(self, event):
		#Determine the column of the cell
		clickXFrac = event.x/self.graDisImgDim[0]
		tempX1 = math.floor(clickXFrac*self.infSamGriDat[1])/self.infSamGriDat[1]
		tempX2 = math.ceil(clickXFrac*self.infSamGriDat[1])/self.infSamGriDat[1]
		col = min([int(tempX1*self.infSamGriDat[1]),self.infSamGriDat[1]-1])
		
		#Determine the row of the cell 
		clickYFrac = event.y/self.graDisImgDim[1]
		tempY1 = math.floor(clickYFrac*self.infSamGriDat[0])/self.infSamGriDat[0]
		tempY2 = math.ceil(clickYFrac*self.infSamGriDat[0])/self.infSamGriDat[0]
		row = min([int(tempY1*self.infSamGriDat[0]),self.infSamGriDat[0]-1])
		
		return [row, col, tempX1, tempY1, tempX2, tempY2]
		
	#Get mouse press location and update the seed sample analysis image.
	def getMousePressLeft(self, event):
		print ("mouse clicked at", event.x, event.y)
		#Stop if the sample has not been graded yet.
		if self.infClaStrVar.get() == "No grade yet.":
			self.infLogTxt.insert(END, "\nNo seed sample graded yet.")
			return
		
		#Determine the coordinates of the display image grid cell at the click location
		#Also determine the image currently being displayed.
		if self.toggleState==0:
			tempImg = iForm.cropImg(self.rawSamImg, self.infSamCroDat[0], self.infSamCroDat[1])
		elif self.toggleState==1:
			tempImg = self.togImgWm
		elif self.toggleState==2:
			tempImg = self.togImgRel
		elif self.toggleState==3:
			tempImg = self.togImgDGR
		elif self.toggleState==4:
			tempImg = self.togImgConf
		
		#Get the row and column of the mouse click.
		self.row, self.col, self.tempX1, self.tempY1, self.tempX2, self.tempY2 = self.getClickRowAndCol(event)
		
		#Appropriately crop the image
		self.exaDisImg = iForm.cropImg(tempImg, (self.tempX1,self.tempY1), (self.tempX2,self.tempY2))
		
		#Update the examination image display
		self.updateLabelImg(self.exaDis, convImg(iForm.resizeImg(self.exaDisImg, self.exaDisImgDim[0], self.exaDisImgDim[1])))
		
		#Update the seed area text
		flattenedIndex = self.row*self.infSamGriDat[1]+self.col
		areaFul = self.seedSampleAreas[flattenedIndex][0]
		areaRel = float(self.seedSampleAreas[flattenedIndex][1])
		areaDGR = self.seedSampleAreas[flattenedIndex][2]
		dgrFrac = self.seedSampleInfo[flattenedIndex][1]
		dgrFracStr = "{:5.3f}%".format(areaDGR/areaRel*100) if areaRel > 0 else 'NaN'
		
		self.exaAnaTxt.config(state=NORMAL)
		self.exaAnaTxt.delete(0., END)
		
		seedAreaText = 'Grade*: ' if str(dgrFrac)[:5]=='cust_' else 'Grade: '		
		seedAreaText = seedAreaText + self.dgrToString(dgrFrac)
		seedAreaText = seedAreaText + "\nRow: " + str(self.row+1)
		seedAreaText = seedAreaText + "\nColumn: " + str(self.col+1)
		seedAreaText = seedAreaText + "\nSeed Pixel Area: " + str(areaFul)
		seedAreaText = seedAreaText + "\nSmear Pixel Area: " + str(areaRel)
		seedAreaText = seedAreaText + "\nDGR Pixel Area: " + str(areaDGR)		
		seedAreaText = seedAreaText + "\nSeed DGR %: " + dgrFracStr
		self.exaAnaTxt.insert(END, seedAreaText)
		self.exaAnaTxt.config(state=DISABLED)
	
	#Return the string representation of the seed fraction
	def dgrToString(self, seedFracStr, char=False, asterisk=False):
		
		stringRep = ''

		if seedFracStr=='-1' or 'NaN' in str(seedFracStr):
			stringRep = 'NaN' if char==False else 'N'
		elif sClas.getSeedAnly().isDGR(seedFracStr):
			stringRep = 'DGR' if char==False else 'D'
		else:
			stringRep = 'Not DGR' if char==False else 'G'
		
		if asterisk and str(seedFracStr)[:5]=='cust_':
			stringRep = stringRep + '*'
		
		return stringRep
	
	#Get mouse press location and update the seed sample analysis image.
	def getMousePressRight(self, event):
		print ("right mouse clicked at", event.x, event.y)
		#Stop if the sample has not been graded yet.
		if self.infClaStrVar.get() == "No grade yet.":
			self.infLogTxt.insert(END, "\nNo seed sample graded yet.")
			return
			
		# display the popup menu
		self.row, self.col, self.tempX1, self.tempY1, self.tempX2, self.tempY2 = self.getClickRowAndCol(event)
		try:
			self.popup.tk_popup(event.x_root, event.y_root, 0)
		finally:
			# make sure to release the grab (Tk 8.0a1 only)
			self.popup.grab_release()
	
	#Set the seed to be DGR
	def setSeedCustDGR(self):
	
		#Modify the DGR rating
		self.seedSampleInfoMod[self.row*self.infSamGriDat[1]+self.col][1] = 'cust_DGR'
		
		#Update the toggle images
		self.printGradeToggleImgs(self.seedSampleInfoMod, toggleState=self.toggleState)
		
	#Set the seed to be not DGR
	def setSeedCustNotDGR(self):
		#Modify the DGR rating
		self.seedSampleInfoMod[self.row*self.infSamGriDat[1]+self.col][1] = 'cust_NotDGR'
		
		#Update the toggle images
		self.printGradeToggleImgs(self.seedSampleInfoMod, toggleState=self.toggleState) 
		
	#Set the seed to be NaN
	def setSeedCustNaN(self):
		#Modify the DGR rating
		self.seedSampleInfoMod[self.row*self.infSamGriDat[1]+self.col][1] = 'cust_NaN'
		
		#Update the toggle images
		self.printGradeToggleImgs(self.seedSampleInfoMod, toggleState=self.toggleState)
		
	#Reset the seed's grade
	def resetSeedGrade(self):
		#Update the toggle images
		self.printGradeToggleImgs(self.seedSampleInfo, toggleState=self.toggleState)
	
#run the program
def runGUI():
	root = Tk()
	application = GUI(master=root)
	application.mainloop()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~Run the program~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
	runGUI()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
