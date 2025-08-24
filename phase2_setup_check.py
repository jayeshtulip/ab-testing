"""
Phase 2 Environment Setup Check
Run this first to verify all dependencies are installed
"""

import sys
import subprocess
import importlib

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version should be 3.8 or higher")
        return False

def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'numpy',
        'scipy',
        'scikit-learn',
        'requests',
        'matplotlib',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(packages):
    """Install missing packages"""
    if not packages:
        print("🎉 All packages are already installed!")
        return
    
    print(f"\n📦 Installing missing packages: {packages}")
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def main():
    print("🔍 Phase 2 Environment Setup Check")
    print("=" * 40)
    
    # Check Python version
    python_ok = check_python_version()
    
    if not python_ok:
        print("\n❌ Please update Python to version 3.8 or higher")
        return
    
    print("\n📦 Checking Required Packages...")
    missing = check_required_packages()
    
    if missing:
        print(f"\n⚠️  Found {len(missing)} missing packages")
        install_choice = input("Install missing packages? (y/n): ").lower().strip()
        
        if install_choice == 'y':
            install_missing_packages(missing)
        else:
            print("❌ Cannot proceed without required packages")
            return
    
    print("\n🎉 Environment Check Complete!")
    print("✅ Ready to run Phase 2 scripts")

if __name__ == "__main__":
    main()