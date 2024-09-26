@echo off
setlocal enabledelayedexpansion

rem Path to your Python script
set PYTHON_SCRIPT=computeLayoutLoop.py

rem Loop from 0 to 23
for /l %%i in (0,1,23) do (
    echo Running with index = %%i
    
    rem Run the Python script with the current index as an argument
    python "%PYTHON_SCRIPT%" --index %%i
    
    echo Finished running with index = %%i
    echo -------------------------
)

echo All iterations completed!

rundll32 user32.dll,MessageBeep


endlocal