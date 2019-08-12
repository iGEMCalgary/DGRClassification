#Imports
import sys
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math

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
from MGM_DGR_SeedClassifier import SeedSampleAnalyzer

#~~~~~~~~~~~~~~~~~~~~~~~Create Backend~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Other classes 
iForm = ImgUtility()
iCamr = ImgCamera(cameraIndex=0)
iCali = ImgCalibrator(iCamr, "CalibrationImages", "guiCalibPhotoTest.png")
sClas = SeedClassifier()
sAnly = SeedSampleAnalyzer()

#Convert an image from numpy format to GUI usable format
convImg = lambda x: ImageTk.PhotoImage(image=Image.fromarray(iForm.convertNpToRGBArr(x)))

#Create a black 3 channel RGB numpy image with the given dimensions (x,y)
blankImg = lambda x: np.zeros((x[0], x[1], 3))

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
		master.title("yOIL Seed Sample Classifier")
		wMin,hMin = 1000, 600
		wMax, hMax = 1100, 720
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
		
		buttonPanel = Canvas(master, width=180, height=hMin, bg="#8ac926")
		buttonPanel.pack(side=LEFT)
		
		#Create button description labels
		leftButLab = Label(buttonPanel, text="Left button takes a photo using the camera", bg="#8ac926").grid(row=0, column=0, columnspan=4, pady=2)
		rightButLab = Label(buttonPanel, text="Right button browses for an image", bg="#8ac926").grid(row=1, column=0, columnspan=4)
		
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
		
		self.graBut = Button(buttonPanel, text='G', width=2, command=self.graButCmd, bg="#1982C4").grid(row=5,column=2, columnspan=2)
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
		self.infClaLabVar = Label(buttonPanel, textvariable=self.infClaStrVar, bg="#FFCA3A").grid(sticky='W',row=18,column=2, columnspan=2)
		self.infClaLab = Label(buttonPanel, text="Grade of Sample:", bg="#8ac926").grid(sticky='E',row=18,column=0, columnspan=2)
		
		self.infLodStrVar = StringVar()
		self.infLodStrVar.set("No grading completed.")
		self.infLodLabVar = Label(buttonPanel, textvariable=self.infLodStrVar, bg="#FFCA3A").grid(sticky='W',row=19,column=2, columnspan=2)
		self.infLodLab = Label(buttonPanel, text="Grading Timing:", bg="#8ac926").grid(sticky='E',row=19,column=0, columnspan=2)
		
		#Text log
		self.infLogTxt = Text(buttonPanel, bg="Grey", width=40, height=16)
		self.infLogTxt.grid(row=20,column=0, columnspan=4)
		self.infLogTxt.insert(END, "Welcome!\nFor reference, Red=DGR, Yellow=Fine, and Blue=Absent.")
		self.infLogTxt.insert(END, "\nAlso, The lighter the green, the higher the confidence.")
		self.infLogTxt.insert(END, "\nAlso, use the keys 'q', 'w', 'e', 'r', and 't' to toggle analysis images once a grading is complete.")		
		
		self.infLogScr = Scrollbar(buttonPanel, command=self.infLogTxt.yview)
		self.infLogScr.grid(row=20, column=4, sticky='nsew')
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
		self.preDisCalImgDim = (200,200)
		self.rawCalImg = blankImg(self.preDisCalImgDim)
		self.calibUpdated = False
		self.preDisCalImg = convImg(iForm.drawBox(blankImg(self.preDisCalImgDim),(0,0),self.preDisCalImgDim))
		self.preDisCal = Label(imagePanel, image=self.preDisCalImg, width=self.preDisCalImgDim[0], height=self.preDisCalImgDim[1])
		self.preDisCal.grid(row=1,column=0, padx=5, pady=5)
		self.preDisCalLab = Label(imagePanel, text="Preview Image for Calibration", bg="#71A520").grid(row=0,column=0, pady=2)
		
		self.preDisGraImgDim = (200,200)
		self.rawSamImg = blankImg(self.preDisGraImgDim)
		self.preDisGraImg = convImg(iForm.drawGrid(blankImg(self.preDisGraImgDim),(0,0),self.preDisGraImgDim,1,1))
		self.preDisGra = Label(imagePanel, image=self.preDisGraImg, width=self.preDisGraImgDim[0], height=self.preDisGraImgDim[1])
		self.preDisGra.grid(row=1,column=1, padx=5, pady=5)
		self.preDisGraLab = Label(imagePanel, text="Preview Image for Seed Sample", bg="#71A520").grid(row=0, column=1)
		
		self.calDisImgDim = (200,200)
		self.calDisImg = convImg(blankImg(self.calDisImgDim))
		self.useCalImg = blankImg(self.calDisImgDim)
		self.calDis = Label(imagePanel, image=self.calDisImg, width=self.calDisImgDim[0], height=self.calDisImgDim[1])
		self.calDis.grid(sticky="NW", row=1,column=2, padx=5, pady=5)
		self.calDisLab = Label(imagePanel, text= "Calibration Image", bg="#71A520").grid(row=0,column=2)
		
		self.graDisImgDim = (400,400)
		self.graDisImg = convImg(iForm.drawGrid(blankImg(self.graDisImgDim),(0,0),self.graDisImgDim,1,1))
		self.sampleUpdated = False
		self.graDis = Label(imagePanel, image=self.graDisImg, width=self.graDisImgDim[0], height=self.graDisImgDim[1])
		self.graDis.grid(row=3,column=0, rowspan=2, columnspan=2, padx=5, pady=5)
		self.graDisLab = Label(imagePanel, text= "Seed Sample Image:", bg="#71A520").grid(sticky='E',row=2, column=0)
		self.graDisLabStrVar = StringVar()
		self.graDisLabStrVar.set("No Image Yet.")
		self.graDisLabVar = Label(imagePanel, textvariable=self.graDisLabStrVar, bg="#FFCA3A").grid(sticky='W',row=2,column=1)
		
		self.exaDisImgDim = (200,200)
		self.exaDisImg = convImg(blankImg(self.exaDisImgDim))
		self.exaDis = Label(imagePanel, image=self.exaDisImg, width=self.exaDisImgDim[0], height=self.exaDisImgDim[1])
		self.exaDis.grid(sticky="NW", row=3,column=2, padx=5, pady=5)
		self.exaDisLab = Label(imagePanel, text= "Examined Seed", bg="#71A520").grid(row=2, column=2)
		
		"""
		self.exaAnaFrame = Frame(imagePanel, width=200, height=200,bg="Blue")
		self.exaAnaFrame.grid(sticky="NW", row=4, column=2, padx=5, pady=5)
		#self.exaAnaStrVar = StringVar()
		#self.exaAnaStrVar.set("balls")
		self.exaAnaTxt = Text(self.exaAnaFrame)
		self.exaAnaTxt.insert(END,"BALLS")
		self.exaAnaTxt.place()
		"""
		self.exaAnaTxt = Text(imagePanel, bg="#FFCA3A", width=24, height=11)
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
		
		master.bind("<Key>", self.setToggleStatus)
		self.graDis.bind("<Button-1>", self.getMousePress)
	
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
		ogCalibrationImg, ogOGImg = iCali.takeCalibrationPhoto(leftTopCorner=self.infCalCroDat[0], rightBotCorner=self.infCalCroDat[1])
		self.rawCalImg = ogOGImg
		self.useCalImg = ogCalibrationImg
		self.updateCalibImage(ogOGImg, ogCalibrationImg)

		
	#Take and display the seed sample image
	def samButCmd(self):
		#Set calibration image
		print("calibration button pressed")
		self.newCamImg = convImg(iCamr.takePhoto())
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
		if len(self.preButFilPath) > 0:
			ogCalibrationImg, ogOGImg = iCali.takeCalibrationPhoto(leftTopCorner=self.infCalCroDat[0], 
										rightBotCorner=self.infCalCroDat[1], filePath=self.preButFilPath)	
			self.rawCalImg=ogOGImg
			self.useCalImg = ogCalibrationImg
			
			self.updateCalibImage(ogOGImg, ogCalibrationImg)
		
	#Browse for and display the seed sample image
	def samButFilCmd(self):
		print("sample browse button pressed")
		tempPath = browseImageFromComp(0)
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
		seedSampleList = iForm.gridPartitionImg(iForm.cropImg(self.rawSamImg, self.infSamCroDat[0], self.infSamCroDat[1]), self.infSamGriDat[0], self.infSamGriDat[1])	
		
		#Classify each seed and associate the data with each seed
		seedSampleInfo, timeStr = sClas.classifySeedSample(seedSampleList, self.useCalImg)		
		self.seedSampleAreas = [seedInfo[2] for seedInfo in seedSampleInfo]
		
		#Analyze the total data of the seed sample
		seedAnalysisString = sAnly.analyzeSeedSample(seedSampleInfo, timeStr)
		self.infLogTxt.insert(END, seedAnalysisString[0])
		self.infClaStrVar.set(seedAnalysisString[0][35:42])
		self.infLodStrVar.set(seedAnalysisString[1])
		
		#Create images for toggling
		self.togImgWm, self.togImgRel, self.togImgDGR, self.togImgConf = sAnly.createInfoImages(seedSampleInfo, self.infSamGriDat[0], self.infSamGriDat[1])
		self.graDisLabStrVar.set("Raw Image")
		self.toggleState=0
	
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
				
		ogCalibrationImg, ogOGImg = iCali.takeCalibrationPhoto(img=self.rawCalImg,leftTopCorner=self.infCalCroDat[0], 
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
			if newX1<0. or newX1>1. or newY1<0. or newY1>1. or newX2<0. or newX2>1. or newY2<0. or newY2>1.:
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
	
	#Get mouse press location and update the seed sample analysis image.
	def getMousePress(self, event):
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
		
		#Appropriately crop the image
		self.exaDisImg = iForm.cropImg(tempImg, (tempX1,tempY1), (tempX2,tempY2))
		
		#Update the examination image display
		self.updateLabelImg(self.exaDis, convImg(iForm.resizeImg(self.exaDisImg, self.exaDisImgDim[0], self.exaDisImgDim[1])))
		
		#Update the seed area text
		areaFul = self.seedSampleAreas[row*self.infSamGriDat[1]+col][0]
		areaRel = self.seedSampleAreas[row*self.infSamGriDat[1]+col][1]
		areaDGR = self.seedSampleAreas[row*self.infSamGriDat[1]+col][2]
		dgrFracStr = "{:5.3f}%".format((areaDGR/areaRel)*100) if areaRel > 0 else 'NaN'
		
		self.exaAnaTxt.config(state=NORMAL)
		self.exaAnaTxt.delete(0., END)
		seedAreaText = "Row: " + str(row+1)
		seedAreaText = seedAreaText + "\nColumn: " + str(col+1)
		seedAreaText = seedAreaText + "\nSeed Pixel Area: " + str(areaFul)
		seedAreaText = seedAreaText + "\nSmear Pixel Area: " + str(areaRel)
		seedAreaText = seedAreaText + "\nDGR Pixel Area: " + str(areaDGR)		
		seedAreaText = seedAreaText + "\nSeed DGR %: " + dgrFracStr
		self.exaAnaTxt.insert(END, seedAreaText)
		self.exaAnaTxt.config(state=DISABLED)
		
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~Run the program~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
root = Tk()
application = GUI(master=root)
application.mainloop()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
