#!/usr/bin/env python3
"""
Dashboard Import Fix Script
Diagnoses and fixes import issues for the Streamlit dashboard
"""

import os
import sys
from pathlib import Path

def diagnose_project_structure():
    """Diagnose project structure and paths"""
    print("🔍 Diagnosing Project Structure")
    print("=" * 40)
    
    current_dir = Path.cwd()
    print(f"Current working directory: {current_dir}")
    
    # Check if we're in the dashboard directory
    if current_dir.name == "dashboard":
        project_root = current_dir.parent
        print(f"✅ In dashboard directory")
        print(f"Project root: {project_root}")
    else:
        print(f"⚠️  Not in dashboard directory")
        # Look for dashboard directory
        dashboard_dir = current_dir / "dashboard"
        if dashboard_dir.exists():
            print(f"Found dashboard directory at: {dashboard_dir}")
            project_root = current_dir
        else:
            print("❌ Cannot find dashboard directory")
            return False, None, None
    
    src_dir = project_root / "src"
    print(f"Expected src directory: {src_dir}")
    print(f"Src exists: {src_dir.exists()}")
    
    return True, project_root, src_dir

def check_src_structure(src_dir):
    """Check src directory structure"""
    print(f"\n🔍 Checking src directory: {src_dir}")
    
    if not src_dir.exists():
        print("❌ src directory does not exist!")
        return False
    
    required_structure = {
        "utils": ["__init__.py", "config.py"],
        "data": ["__init__.py", "database.py", "models.py"],
        "agents": ["__init__.py", "base_agent.py", "research_agent.py", "analysis_agent.py", "fact_checker_agent.py"]
    }
    
    missing_items = []
    
    for dir_name, files in required_structure.items():
        dir_path = src_dir / dir_name
        print(f"\n📁 {dir_name}/ directory:")
        
        if not dir_path.exists():
            print(f"❌ Directory missing: {dir_path}")
            missing_items.append(f"{dir_name}/ directory")
            continue
        else:
            print(f"✅ Directory exists: {dir_path}")
        
        for file_name in files:
            file_path = dir_path / file_name
            if file_path.exists():
                print(f"  ✅ {file_name}")
            else:
                print(f"  ❌ {file_name}")
                missing_items.append(f"{dir_name}/{file_name}")
    
    if missing_items:
        print(f"\n❌ Missing items:")
        for item in missing_items:
            print(f"   - {item}")
        return False
    
    print(f"\n✅ All required files found in src/")
    return True

def test_imports(src_dir):
    """Test imports from src directory"""
    print(f"\n🧪 Testing imports from: {src_dir}")
    
    # Add src to path
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
        print(f"✅ Added {src_str} to Python path")
    
    # Test individual imports
    test_imports_list = [
        ("utils.config", "config"),
        ("data.database", "initialize_database"),
        ("data.models", "Stock"),
        ("agents.base_agent", "BaseAgent"),
        ("agents.research_agent", "ResearchAgent"),
        ("agents.analysis_agent", "AnalysisAgent"),
        ("agents.fact_checker_agent", "FactCheckerAgent")
    ]
    
    successful_imports = []
    failed_imports = []
    
    for module_name, item_name in test_imports_list:
        try:
            module = __import__(module_name, fromlist=[item_name])
            getattr(module, item_name)
            print(f"✅ {module_name}.{item_name}")
            successful_imports.append(module_name)
        except ImportError as e:
            print(f"❌ {module_name}.{item_name}: {e}")
            failed_imports.append((module_name, str(e)))
        except AttributeError as e:
            print(f"⚠️  {module_name}.{item_name}: {e}")
            failed_imports.append((module_name, str(e)))
    
    print(f"\n📊 Import Results:")
    print(f"   ✅ Successful: {len(successful_imports)}")
    print(f"   ❌ Failed: {len(failed_imports)}")
    
    if failed_imports:
        print(f"\n❌ Failed imports:")
        for module, error in failed_imports:
            print(f"   - {module}: {error}")
        return False
    
    return True

def create_missing_init_files(src_dir):
    """Create missing __init__.py files"""
    print(f"\n🔧 Creating missing __init__.py files...")
    
    subdirs = ["utils", "data", "agents"]
    created_files = []
    
    for subdir in subdirs:
        dir_path = src_dir / subdir
        if dir_path.exists():
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                try:
                    init_file.touch()
                    print(f"✅ Created: {init_file}")
                    created_files.append(str(init_file))
                except Exception as e:
                    print(f"❌ Failed to create {init_file}: {e}")
    
    if created_files:
        print(f"📝 Created {len(created_files)} __init__.py files")
    else:
        print(f"ℹ️  All __init__.py files already exist")
    
    return len(created_files) > 0

def generate_launch_command(project_root):
    """Generate the correct launch command"""
    print(f"\n🚀 Recommended Launch Commands:")
    print("=" * 40)
    
    dashboard_dir = project_root / "dashboard"
    
    print(f"Option 1 (From dashboard directory):")
    print(f"  cd {dashboard_dir}")
    print(f"  python run_dashboard.py")
    
    print(f"\nOption 2 (Manual launch):")
    print(f"  cd {dashboard_dir}")
    print(f"  PYTHONPATH='{project_root}/src' streamlit run streamlit_app.py")
    
    print(f"\nOption 3 (From project root):")
    print(f"  cd {project_root}")
    print(f"  PYTHONPATH='./src' streamlit run dashboard/streamlit_app.py")

def main():
    """Main diagnostic function"""
    print("🩺 Dashboard Import Diagnostics")
    print("=" * 50)
    
    # Step 1: Diagnose project structure
    success, project_root, src_dir = diagnose_project_structure()
    if not success:
        print("\n❌ Cannot continue - project structure issues")
        return False
    
    # Step 2: Check src structure
    if not check_src_structure(src_dir):
        print("\n🔧 Attempting to fix missing __init__.py files...")
        create_missing_init_files(src_dir)
        
        # Recheck after fix attempt
        if not check_src_structure(src_dir):
            print("\n❌ Cannot continue - missing critical files in src/")
            print("\n💡 Please ensure you have all the required files:")
            print("   - Run the main system first: cd src && python main.py simple")
            print("   - Make sure all agents are properly created")
            return False
    
    # Step 3: Test imports
    if not test_imports(src_dir):
        print("\n❌ Import test failed")
        print("\n💡 Possible solutions:")
        print("   1. Run: cd src && python main.py simple")
        print("   2. Check for any syntax errors in the Python files")
        print("   3. Ensure all dependencies are installed: pip install -r requirements.txt")
        return False
    
    # Step 4: Generate launch commands
    generate_launch_command(project_root)
    
    print(f"\n✅ All diagnostics passed!")
    print(f"🚀 Your dashboard should now work correctly!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 Diagnostics completed successfully!")
        print(f"   Run the launch commands above to start your dashboard")
    else:
        print(f"\n💥 Diagnostics failed!")
        print(f"   Please fix the issues above before launching the dashboard")
    
    sys.exit(0 if success else 1)