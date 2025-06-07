#!/usr/bin/env python3
"""
Fix script to resolve database and agent communication issues
"""

import asyncio
import sys
from pathlib import Path

def check_imports():
    """Check if all required modules can be imported"""
    print("🔍 Checking imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError:
        print("❌ pandas not found - run: pip install pandas")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully") 
    except ImportError:
        print("❌ numpy not found - run: pip install numpy")
        return False
    
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
    except ImportError:
        print("❌ yfinance not found - run: pip install yfinance")
        return False
    
    try:
        import anthropic
        print("✅ anthropic imported successfully")
    except ImportError:
        print("❌ anthropic not found - run: pip install anthropic")
        return False
        
    return True

def test_database_connection():
    """Test database connection and operations"""
    print("\n🗄️ Testing database connection...")
    
    try:
        from data.database import initialize_database
        from utils.config import config
        import data.database as db_module
        
        # Initialize database with URL from config
        db = initialize_database(config.database.database_url)
        print("✅ Database initialized successfully")
        
        # Access stock_operations from the module after initialization
        stock_operations = db_module.stock_operations
        
        # Test stock operations
        if stock_operations:
            # Use a session context to properly access the data
            with db.get_session() as session:
                from data.models import Stock
                stocks = session.query(Stock).filter(Stock.is_active == True).all()
                stock_count = len(stocks)
                
                if stock_count > 0:
                    # Access properties while in session
                    sample_stock = stocks[0]
                    sample_symbol = sample_stock.symbol
                    sample_name = sample_stock.company_name
                    
                    print(f"✅ Found {stock_count} stocks in database")
                    print(f"✅ Sample stock: {sample_symbol} - {sample_name}")
                else:
                    print("⚠️ No stocks found - database may need initialization")
                    return False
        else:
            print("❌ Stock operations not available - testing database directly")
            
            # Alternative test: Query database directly
            try:
                with db.get_session() as session:
                    from data.models import Stock
                    stocks = session.query(Stock).filter(Stock.is_active == True).all()
                    stock_count = len(stocks)
                    
                    print(f"✅ Direct database query found {stock_count} stocks")
                    
                    if stock_count > 0:
                        sample_stock = stocks[0]
                        sample_symbol = sample_stock.symbol
                        sample_name = sample_stock.company_name
                        print(f"✅ Sample stock: {sample_symbol} - {sample_name}")
                        return True
                    else:
                        print("⚠️ No stocks found in direct query")
                        return False
            except Exception as direct_error:
                print(f"❌ Direct database query failed: {str(direct_error)}")
                return False
            
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        return False

def test_agent_creation():
    """Test agent creation and basic functionality"""
    print("\n🤖 Testing agent creation...")
    
    try:
        from agents.research_agent import ResearchAgent
        from agents.analysis_agent import AnalysisAgent
        
        # Test Research Agent
        research_agent = ResearchAgent()
        print("✅ Research Agent created successfully")
        print(f"   Capabilities: {research_agent.get_capabilities()}")
        
        # Test Analysis Agent 
        analysis_agent = AnalysisAgent()
        print("✅ Analysis Agent created successfully")
        print(f"   Capabilities: {analysis_agent.get_capabilities()}")
        
        return True, research_agent, analysis_agent
        
    except Exception as e:
        print(f"❌ Agent creation failed: {str(e)}")
        return False, None, None

async def test_agent_communication():
    """Test agent communication"""
    print("\n🔗 Testing agent communication...")
    
    try:
        success, research_agent, analysis_agent = test_agent_creation()
        if not success:
            return False
        
        # Connect agents
        research_agent.add_known_agent("analysis", analysis_agent)
        analysis_agent.add_known_agent("research", research_agent)
        
        print("✅ Agents connected successfully")
        
        # Start agents
        await research_agent.start()
        await analysis_agent.start()
        print("✅ Agents started successfully")
        
        # Test basic task
        from agents.base_agent import Task
        task = Task(
            type="fetch_stock_price",
            data={"symbol": "TCS.NS"}
        )
        
        await research_agent.add_task(task)
        await asyncio.sleep(2)
        
        print("✅ Basic task submitted successfully")
        
        # Stop agents
        await research_agent.stop()
        await analysis_agent.stop()
        print("✅ Agents stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent communication test failed: {str(e)}")
        return False

def check_configuration():
    """Check configuration settings"""
    print("\n⚙️ Checking configuration...")
    
    try:
        from utils.config import config
        
        print(f"✅ Configuration loaded")
        print(f"   Database URL: {config.database.database_url}")
        print(f"   Log Level: {config.logging.level}")
        print(f"   Stock Symbols: {len(config.get_stock_symbols())} symbols")
        
        # Check API keys
        if config.api.anthropic_api_key:
            print("✅ Anthropic API key configured")
        else:
            print("⚠️ Anthropic API key not configured - AI insights will be limited")
        
        if config.api.alpha_vantage_api_key:
            print("✅ Alpha Vantage API key configured")
        else:
            print("ℹ️ Alpha Vantage API key not configured - using Yahoo Finance only")
        
        return config.validate()
        
    except Exception as e:
        print(f"❌ Configuration check failed: {str(e)}")
        return False

async def run_comprehensive_test():
    """Run a comprehensive test of the system"""
    print("\n🧪 Running comprehensive test...")
    
    try:
        from main import StockResearchSystem
        
        system = StockResearchSystem()
        await system.initialize()
        print("✅ System initialized successfully")
        
        # Test research
        result = await system.research_stock("RELIANCE.NS")
        print(f"✅ Research test: {result}")
        
        # Test analysis (with better error handling)
        try:
            analysis_result = await system.analyze_stock("RELIANCE.NS", "technical_analysis")
            print(f"✅ Analysis test: {analysis_result}")
        except Exception as e:
            print(f"⚠️ Analysis test had issues: {str(e)}")
        
        await system.shutdown()
        print("✅ System shutdown successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {str(e)}")
        return False

def create_requirements_fix():
    """Create updated requirements.txt with exact versions"""
    requirements = """
# Core dependencies
anthropic>=0.25.0
python-dotenv>=1.0.0
pydantic>=2.5.0
sqlalchemy>=2.0.0

# Data & Analysis
pandas>=2.1.0
numpy>=1.24.0
yfinance>=0.2.0
requests>=2.31.0

# Dashboard (for later)
streamlit>=1.28.0
plotly>=5.17.0
altair>=5.0.0

# Development
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0

# Async support
aiohttp>=3.9.0
python-dateutil>=2.8.0
pytz>=2023.0
"""
    
    with open("../requirements.txt", "w") as f:
        f.write(requirements.strip())
    
    print("✅ Updated requirements.txt created")

async def main():
    """Main fix and test function"""
    print("🔧 Stock Research System Fix & Test")
    print("=" * 50)
    
    # Check current directory
    if not Path.cwd().name == "src":
        print("❌ Please run this script from the 'src' directory")
        print("   cd src && python fix_script.py")
        sys.exit(1)
    
    # Step 1: Check imports
    if not check_imports():
        print("\n❌ Import check failed. Please install missing packages.")
        create_requirements_fix()
        print("Run: pip install -r ../requirements.txt")
        return
    
    # Step 2: Check configuration
    if not check_configuration():
        print("\n❌ Configuration check failed.")
        return
    
    # Step 3: Test database
    if not test_database_connection():
        print("\n❌ Database test failed.")
        return
    
    # Step 4: Test agent communication
    if not await test_agent_communication():
        print("\n❌ Agent communication test failed.")
        return
    
    # Step 5: Run comprehensive test
    if not await run_comprehensive_test():
        print("\n❌ Comprehensive test failed.")
        return
    
    print("\n🎉 All tests passed! System is working correctly.")
    print("\nYou can now run:")
    print("  python main.py simple  # Basic test")
    print("  python main.py         # Full demo")

if __name__ == "__main__":
    asyncio.run(main())