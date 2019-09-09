# DGRClassification

> To run the GUI with the python shell open:
Python MGM_DGR_GUI.py

> To run the GUI standalone double click the MGM_DGR_Main.py file.

> To run the classifier, run the command:
Python MGM_DGR_Classifier.py {Folder name of calibration images} {File name of DGR calibration image} {Folder name of seed sample image} {Number of rows in the seed sample} {Number of columns in the seed sample} {(N)th latest file (you wish to load)}  

> For example:
Python MGM_DGR_Classifier.py CalibrationImages DGRChip.png Images 7 17 0

> To create/update the standalone executable folder and files:
> *Only works with Windows machines.
> *Requires pyInstaller (install with: pip install PyInstaller)
> *Requires pyWin32 (install wit:h pip install pywin32)
pyinstaller --onefile MGM_DGR_GUI.py

> Once EXE successfully built, move 'build', 'dist', 'MGM_DGR_Main.spec'
> folders into the 'Standalone Version' folder, replacing files if needed.
> Update the shortcut 'GreatGrader' for the new MGM_DGR_Main.exe file.