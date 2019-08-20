#Imports
import sys
from sklearn.preprocessing import MinMaxScaler
from MGM_DGR_ImageFormatting import ImgUtility
from MGM_DGR_ImageFormatting import ImgCalibrator
from MGM_DGR_SeedClassifier import SeedClassifier
from MGM_DGR_SeedClassifier import SeedSampleAnalyzer

#~~~~~~~~~~~~~~~~~~~~~~~~Step 0: Define Variables~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
iForm = ImgUtility()						#Used for image formatting
sClas = SeedClassifier()					#Used for seed classification
sAnly = SeedSampleAnalyzer()				#Used for seed sample analysis

#Default program variables
progVariables = [r"CalibrationImages", r"DGRChip.png", r"Images", (1,1), 0]
if len(sys.argv) == 7:
	print("Custom settings used.")
	progVariables = sys.argv[1:4]
	progVariables.append((int(sys.argv[4]), int(sys.argv[5]))) 
	progVariables.append(int(sys.argv[6]))
else:
	print("Default settings used.")

calibFolderPath = 	progVariables[0]		#Folder for calibration images
calibFilePath =		progVariables[1]				#DGR calibration chip image
seedFolderPath = 	progVariables[2]				#Folder for input seed images
seedSampleRows, seedSampleCols = progVariables[3]		#Number of rows and cols in the sample img 
startOffset =		progVariables[4]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~Step 1: Format Input Images~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
seedSampleImg = iForm.readLatestImageFromFolder(seedFolderPath, startOffset=startOffset)#Read the sample image with the multiple seeds
seedSampleList = iForm.gridPartitionImg(seedSampleImg, seedSampleRows, seedSampleCols)	#Divide the whole image into the specified individual seed images
for seeds in seedSampleList:
	iForm.displayImg(seeds)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~Step 2: Calibration~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dgrCalibChip = iForm.readImageFromFolder(calibFolderPath, calibFilePath)				#Read the calibration chip
dgrCalibChip = iForm.averageImg(dgrCalibChip)											#Average out the calibration chip to get the proper calibration values
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~Step 3: Classify Images~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
seedSampleInfo = sClas.classifySeedSample(seedSampleList, dgrCalibChip)[0]					#Classify each seed and associate the data with each seed
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~Step 4: Produce Analysis~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sAnly.analyzeSeedSample(seedSampleInfo)					#Analyze the total data of the seed sample
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

