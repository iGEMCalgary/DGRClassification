# DGRClassification

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

> To run the GUI with the python shell open:
Python MGM_DGR_GUI.py

> To run the GUI standalone double click the MGM_DGR_Main.py file.
 > To enable camera functionality through an android phone, download IP Webcam on a phone and "Start server".
 > Then, copy the ip address into the appropriate entry and press the "Activate IP Webcam Phone Camera" button.

> To create/update the standalone executable folder and files:
> *Only works with Windows machines.
> *Requires pyInstaller (install with: pip install PyInstaller)
> *Requires pyWin32 (install with: pip install pywin32)
pyinstaller --onefile MGM_DGR_GUI.py

> Once EXE successfully built, move 'build' and 'dist' folders and the 'MGM_DGR_Main.spec' 
> file into the 'Great Grader' folder in the GreatGrader 
> (https://github.com/iGEMCalgary/GreatGrader.git) repo, replacing files if needed.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

> To run the console script classifier, run the command:
Python MGM_DGR_Classifier.py {Folder name of calibration images} {File name of DGR calibration image} 
			     {Folder name of seed sample image}  {Number of rows in the seed sample} 
			     {Number of columns in the seed sample} {(N)th latest file (you wish to load)}  

> For example:
Python MGM_DGR_Classifier.py CalibrationImages DGRChip.png Images 7 17 0

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~