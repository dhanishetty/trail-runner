@echo off
rem Builds Trail-Runner.exe (folder build\dist\Trail-Runner). Use a SHORT path for the build folder: Windows limits paths to 260 characters.
rem Needs a clean venv with only requirements.txt + pyinstaller, and pose_landmark_lite.tflite inside its mediapipe\modules\pose_landmark folder
rem (it is downloaded the first time camera control runs from a normal Python install).
set B=C:\trb
%B%\venv\Scripts\python.exe -m PyInstaller --noconfirm --windowed --name Trail-Runner --distpath %B%\dist --workpath %B%\build --specpath %B% --paths "%~dp0." --add-data "%~dp0assets;assets" --add-data "%~dp0models;models" --collect-data ursina --collect-all panda3d --collect-all mediapipe --hidden-import pose_control --exclude-module IPython --exclude-module tkinter --exclude-module jax --exclude-module jaxlib --exclude-module pytest --exclude-module sphinx "%~dp0wheel_ursina.py"
copy "%~dp0track_layout.txt" %B%\dist\Trail-Runner\ >nul
