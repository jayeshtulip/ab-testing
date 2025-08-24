@echo off
mkdir phase3b_automation
cd phase3b_automation
mkdir core api tests config
type nul > core\__init__.py
type nul > api\__init__.py
type nul > tests\__init__.py
type nul > config\__init__.py
echo Phase 3B directories created!
echo Now copy the Python files from Claude into these directories.
pause