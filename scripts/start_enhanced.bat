@echo off
echo  Starting Enhanced A/B Testing API
echo ===================================

cd /d "%~dp0\.."
set PYTHONPATH=%cd%\src

echo Starting enhanced server...
python src\ab_testing\ab_testing_api_enhanced.py
