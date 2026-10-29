@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -m ocg.run --config configs\synthetic_abrupt.yaml --output results\synthetic_abrupt_local.json --device cpu
set "run_status=%errorlevel%"
if not "%run_status%"=="0" echo Experiment failed with exit code %run_status%.
if "%run_status%"=="0" echo Results saved to results\synthetic_abrupt_local.json
pause
exit /b %run_status%
