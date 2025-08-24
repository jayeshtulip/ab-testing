@echo off
echo  Running A/B Testing Tests
echo ============================

set PYTHONPATH=%~dp0\..\src

python "%~dp0\test_ab_pipeline.py"

pause
