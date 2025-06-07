#!/usr/bin/env python3
"""
Quick test to verify database path and access
"""

import os
import sys
from pathlib import Path

def test_database_path():
    """Test if database exists and is accessible"""
    print("🔍 Database Path Test")
    print("=" * 30)
    
    # Current directory info
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Detect project structure
    if current_dir.name == "dashboard":
        project_root = current_dir.parent
        print("✅ Running from dashboard directory")
    else:
        project_root = current_dir
        print("ℹ️  Running from project root")
    
    print(f"Project root: {project_root}")
    
    # Check data directory
    data_dir = project_root / "data"
    print(f"Data directory: {data_dir}")
    print(f"Data directory exists: {data_dir.exists()}")
    
    if data_dir.exists():
        files = list(data_dir.glob("*"))
        print(f"Data directory contents: {[f.name for f in files]}")
    
    # Check database file
    db_file = data_dir / "stock_research.db"
    print(f"Database file: {db_file}")
    print(f"Database file exists: {db_file.exists()}")
    
    if db_file.exists():
        stat = db_file.stat()
        print(f"Database file size: {stat.st_size} bytes")
        print(f"Database file permissions: {oct(stat.st_mode)}")
    
    # Test database connection
    if db_file.exists():
        print("\n🧪 Testing database connection...")
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"✅ Database tables: {[t[0] for t in tables]}")
            
            # Check stocks count
            cursor.execute("SELECT COUNT(*) FROM stocks;")
            count = cursor.fetchone()[0]
            print(f"✅ Stocks in database: {count}")
            
            conn.close()
            print("✅ Database connection successful!")
            return True
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    else:
        print("❌ Database file not found")
        return False

def test_src_imports():
    """Test if we can import from src"""
    print("\n🔧 Testing src imports...")
    
    current_dir = Path.cwd()
    if current_dir.name == "dashboard":
        project_root = current_dir.parent
    else:
        project_root = current_dir
    
    src_dir = project_root / "src"
    
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        print(f"✅ Added {src_dir} to Python path")
    
    try:
        from utils.config import config
        print(f"✅ Config loaded successfully")
        print(f"✅ Database URL from config: {config.database.database_url}")
        return True
    except Exception as e:
        print(f"❌ Failed to import config: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Database and Import Test")
    print("=" * 40)
    
    # Test database path
    db_ok = test_database_path()
    
    # Test imports
    import_ok = test_src_imports()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 20)
    print(f"Database access: {'✅ OK' if db_ok else '❌ FAIL'}")
    print(f"Imports: {'✅ OK' if import_ok else '❌ FAIL'}")
    
    if db_ok and import_ok:
        print("\n🎉 All tests passed! Dashboard should work.")
    else:
        print("\n💥 Some tests failed. Check the issues above.")
        
        if not db_ok:
            print("\n💡 To fix database issue:")
            print("   cd ../src && python main.py simple")
        
        if not import_ok:
            print("\n💡 To fix import issue:")
            print("   Ensure you're in the dashboard directory")
            print("   Check that src/ directory exists in parent folder")

if __name__ == "__main__":
    main()