#!/usr/bin/env python3
"""
Setup script for the Streamlit Dashboard
"""

import os
import sys
from pathlib import Path
import subprocess

def create_dashboard_structure():
    """Create dashboard directory structure"""
    print("📁 Creating dashboard structure...")
    
    current_dir = Path.cwd()
    dashboard_dir = current_dir / "dashboard"
    components_dir = dashboard_dir / "components"
    
    # Create directories
    dashboard_dir.mkdir(exist_ok=True)
    components_dir.mkdir(exist_ok=True)
    
    print(f"✅ Dashboard directory: {dashboard_dir}")
    print(f"✅ Components directory: {components_dir}")
    
    return dashboard_dir

def create_dashboard_files(dashboard_dir):
    """Create necessary dashboard files"""
    print("📄 Creating dashboard files...")
    
    # Files are already created via artifacts, just verify they exist
    files_to_check = [
        "streamlit_app.py",
        "run_dashboard.py", 
        "utils.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_name in files_to_check:
        file_path = dashboard_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} missing")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {missing_files}")
        print("Please copy the files from the artifacts to the dashboard directory")
        return False
    
    return True

def install_dashboard_dependencies():
    """Install dashboard-specific dependencies"""
    print("📦 Installing dashboard dependencies...")
    
    dashboard_deps = [
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "altair>=5.0.0"
    ]
    
    try:
        for dep in dashboard_deps:
            print(f"Installing {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep
            ], stdout=subprocess.DEVNULL)
        
        print("✅ Dashboard dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def verify_system_setup():
    """Verify the main system is set up"""
    print("🔍 Verifying system setup...")
    
    # Check for src directory
    src_dir = Path.cwd() / "src"
    if not src_dir.exists():
        print("❌ src directory not found")
        return False
    
    # Check for key files
    key_files = [
        "src/main.py",
        "src/agents/research_agent.py",
        "src/agents/analysis_agent.py", 
        "src/agents/fact_checker_agent.py",
        "src/data/database.py"
    ]
    
    missing_files = []
    for file_path in key_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing system files: {missing_files}")
        return False
    
    # Check if database exists
    data_dir = Path.cwd() / "data"
    if data_dir.exists():
        print("✅ Data directory found")
    else:
        print("⚠️ Data directory not found - will be created on first run")
    
    return True

def test_database_connection():
    """Test database connection"""
    print("🗄️ Testing database connection...")
    
    try:
        # Add src to path
        src_path = str(Path.cwd() / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from data.database import initialize_database
        from utils.config import config
        
        # Try to initialize database
        db_manager = initialize_database(config.database.database_url)
        if db_manager:
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection failed")
            return False
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        print("💡 Run 'cd src && python main.py simple' first to initialize the system")
        return False

def create_launch_instructions(dashboard_dir):
    """Create launch instructions"""
    instructions = f"""
🚀 Dashboard Setup Complete!

📂 Dashboard Location: {dashboard_dir}

🎯 To Launch the Dashboard:

Method 1 (Recommended):
  cd dashboard
  python run_dashboard.py

Method 2 (Manual):
  cd dashboard  
  streamlit run streamlit_app.py

Method 3 (With custom port):
  cd dashboard
  streamlit run streamlit_app.py --server.port 8502

📈 The dashboard will open at: http://localhost:8501

🔧 If you encounter issues:
1. Make sure the main system is working: cd src && python main.py simple
2. Check all dependencies are installed: pip install -r requirements.txt
3. Ensure you have some data: cd src && python main.py

📊 Dashboard Features:
- 📈 Real-time stock charts and analysis
- 🤖 Agent status monitoring  
- 📋 Research reports and insights
- ✅ Data quality validation
- 💼 Portfolio management

Enjoy your Stock Research Dashboard! 🎊
"""
    
    print(instructions)
    
    # Save instructions to file
    instructions_file = dashboard_dir / "LAUNCH_INSTRUCTIONS.md"
    with open(instructions_file, "w") as f:
        f.write(instructions)
    
    print(f"💾 Instructions saved to: {instructions_file}")

def main():
    """Main setup function"""
    print("🚀 Stock Research Dashboard Setup")
    print("=" * 50)
    
    # Step 1: Create directory structure
    dashboard_dir = create_dashboard_structure()
    
    # Step 2: Verify dashboard files exist
    if not create_dashboard_files(dashboard_dir):
        print("\n❌ Setup failed: Missing dashboard files")
        return False
    
    # Step 3: Install dependencies
    if not install_dashboard_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        return False
    
    # Step 4: Verify system setup
    if not verify_system_setup():
        print("\n❌ Setup failed: System not properly configured")
        print("💡 Please run the main system setup first")
        return False
    
    # Step 5: Test database (optional)
    test_database_connection()
    
    # Step 6: Create launch instructions
    create_launch_instructions(dashboard_dir)
    
    print("\n🎉 Dashboard setup completed successfully!")
    print("\n🚀 Ready to launch! Run:")
    print("   cd dashboard")
    print("   python run_dashboard.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)