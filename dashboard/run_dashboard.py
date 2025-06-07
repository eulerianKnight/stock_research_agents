#!/usr/bin/env python3
"""
Dashboard runner script
"""

import subprocess
import sys
import os
from pathlib import Path

def check_streamlit():
    """Check if streamlit is installed"""
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} found")
        return True
    except ImportError:
        print("❌ Streamlit not found")
        return False

def check_project_structure():
    """Check if project structure is correct"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    src_dir = project_root / "src"
    
    print(f"Dashboard dir: {current_dir}")
    print(f"Project root: {project_root}")
    print(f"Expected src dir: {src_dir}")
    
    # Check if src directory exists
    if not src_dir.exists():
        print(f"❌ src directory not found at: {src_dir}")
        return False
    
    # Check key files in src
    required_files = [
        "utils/__init__.py",
        "utils/config.py", 
        "data/__init__.py",
        "data/database.py",
        "agents/__init__.py",
        "agents/base_agent.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = src_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing required files in src/:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def check_dependencies():
    """Check if all dependencies are available"""
    dependencies = [
        ('plotly', 'plotly'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sqlalchemy', 'sqlalchemy')
    ]
    
    missing = []
    for module_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {module_name} found")
        except ImportError:
            print(f"❌ {module_name} not found")
            missing.append(module_name)
    
    return len(missing) == 0, missing

def setup_environment():
    """Setup environment variables and paths"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    src_dir = project_root / "src"
    
    # Set PYTHONPATH to include src directory
    pythonpath = os.environ.get('PYTHONPATH', '')
    src_str = str(src_dir)
    
    if src_str not in pythonpath:
        if pythonpath:
            new_pythonpath = f"{src_str}{os.pathsep}{pythonpath}"
        else:
            new_pythonpath = src_str
        
        os.environ['PYTHONPATH'] = new_pythonpath
        print(f"✅ Set PYTHONPATH to include: {src_str}")
    
    return True

def run_dashboard():
    """Run the Streamlit dashboard"""
    current_dir = Path(__file__).parent
    dashboard_file = current_dir / "streamlit_app.py"
    
    if not dashboard_file.exists():
        print(f"❌ Dashboard file not found: {dashboard_file}")
        return False
    
    print("🚀 Starting Streamlit dashboard...")
    print("📈 Stock Research Multi-Agent System Dashboard")
    print("=" * 50)
    print(f"📂 Working directory: {current_dir}")
    print(f"📄 Dashboard file: {dashboard_file}")
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print("=" * 50)
    
    # Change to dashboard directory
    os.chdir(current_dir)
    
    # Run streamlit with environment setup
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(dashboard_file),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return True
    
    return True

def main():
    """Main runner function"""
    print("🔧 Stock Research Dashboard Launcher")
    print("=" * 40)
    
    # Check project structure first
    if not check_project_structure():
        print("\n❌ Project structure check failed!")
        print("\n💡 Make sure you have:")
        print("   - A 'src' directory in the parent folder")
        print("   - All required Python files in src/")
        print("   - Proper __init__.py files in subdirectories")
        return False
    
    # Check streamlit
    if not check_streamlit():
        print("\n📦 Installing Streamlit...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("✅ Streamlit installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Streamlit")
            return False
    
    # Check other dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"\n📦 Installing missing dependencies: {', '.join(missing)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Run dashboard
    return run_dashboard()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)