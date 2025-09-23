#!/usr/bin/env python3
"""
MindTrack Setup Script
This script helps set up the MindTrack environment and dependencies.
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Please use Python 3.9 or higher.")
        return False

def check_venv():
    """Check if we're in a virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("✅ Running in virtual environment")
        return True
    else:
        print("⚠️ Not running in virtual environment. Highly recommended to use one.")
        return False

def install_requirements():
    """Install Python requirements."""
    if os.path.exists('requirements.txt'):
        return run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing Python dependencies"
        )
    else:
        print("❌ requirements.txt not found")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        'model/saved_model',
        '.streamlit'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")
    
    return True

def setup_secrets():
    """Set up secrets template."""
    secrets_template = '.streamlit/secrets_template.toml'
    secrets_file = '.streamlit/secrets.toml'
    
    if os.path.exists(secrets_template) and not os.path.exists(secrets_file):
        import shutil
        shutil.copy(secrets_template, secrets_file)
        print(f"✅ Created {secrets_file} from template")
        print("📝 Please edit .streamlit/secrets.toml with your Reddit API credentials")
    elif os.path.exists(secrets_file):
        print(f"📁 {secrets_file} already exists")
    else:
        print(f"⚠️ {secrets_template} not found")
    
    return True

def test_imports():
    """Test if all required packages can be imported."""
    packages = [
        'streamlit',
        'torch',
        'transformers',
        'sklearn',
        'praw',
        'lime',
        'pandas',
        'numpy'
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Try running: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are available")
        return True

def main():
    """Main setup function."""
    print("🧠 MindTrack Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check virtual environment
    check_venv()
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Failed to install requirements. Please check the error messages above.")
        sys.exit(1)
    
    # Set up secrets
    setup_secrets()
    
    # Test imports
    if not test_imports():
        print("\n❌ Some packages are missing. Please run 'pip install -r requirements.txt' manually.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. (Optional) Edit .streamlit/secrets.toml with your Reddit API credentials")
    print("2. (Optional) Train the model: python model/train.py")
    print("3. Run the app: streamlit run app/app.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()