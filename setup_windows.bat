@echo off
setlocal
REM One-time setup for split_commercials.py on Windows.
REM Installs Python and ffmpeg with winget if they're missing, then the Python packages.

set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY (
  python -c "import sys" >nul 2>nul && set "PY=python"
)

if not defined PY (
  echo Python not found - installing Python 3.12 with winget...
  winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
  echo.
  echo Python installed. CLOSE this window, open a new one, and run setup_windows.bat again.
  pause
  exit /b
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg not found - installing with winget...
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
  echo.
  echo ffmpeg installed. CLOSE this window, open a new one, and run setup_windows.bat again.
  pause
  exit /b
)

echo Using: %PY%
%PY% -m pip install --upgrade pip
%PY% -m pip install -r "%~dp0requirements.txt"

echo.
echo Setup done. Run it like this (note: "py", not "python"):
echo   py split_commercials.py "D:\Tapes\50s ads.mp4" -o "D:\NeXroll\Prerolls\50s Ads"
pause
