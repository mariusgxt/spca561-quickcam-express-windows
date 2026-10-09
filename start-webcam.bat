@echo off
rem Starts the QuickCam Express as "OBS Virtual Camera". Close this window to stop.
cd /d "%~dp0"
python -m spca561cam %*
if errorlevel 1 pause
