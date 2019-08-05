#Imports
import sys
from MGM_DGR_ImageFormatting import ImgUtility

#~~~~~~~~~~~~~~~~~~~~~~~~Step 0: Define Variables~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
iForm = ImgUtility()						#Used for image formatting

#Command: Python MGM_DGR_ImageCropping.py {loadFolderPath} {loadImgPath} 
#{saveFolderPath} {saveImgPath} {leftCorner} {topCorner} {rightCorner} {botCorner}
assert(len(sys.argv)==9)
loadFolderPath, loadImgPath, saveFolderPath, saveImgPath = sys.argv[1:5]
leftCorner, topCorner, rightCorner, botCorner = sys.argv[5:]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~Step 1: Read the Image~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
img = iForm.readImageFromFolder(loadFolderPath, loadImgPath)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~Step 2: Crop the Image~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
croppedImg = iForm.cropImg(img, (leftCorner, topCorner), (rightCorner, botCorner))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~Step 3: Save the Image~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
iForm.saveImageToFolder(saveFolderPath, saveImgPath, croppedImg)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~