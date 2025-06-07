# src/main.py

import asyncio
import logging
from datetime import datetime
import json
from pathlib import Path

try:
    from utils.config import config
    from data.database import initialize_database, stock_operations
    from agents.base_agent import Task
    from agents.research_agent import ResearchAgent
    from agents.analysis_agent import AnalysisAgent
    from agents.fact_checker_agent import FactCheckerAgent
except ImportError:
    # If running as package, use relative imports
    from .utils.config import config
    from .data.database import initialize_database, stock_operations
    from .agents.base_agent import Task
    from .agents.research_agent import ResearchAgent
    from .agents.analysis_agent import AnalysisAgent
    from .agents.fact_checker_agent import FactCheckerAgent

class StockResearchSystem:
    """Main orchestrator for the stock research system"""
    
    def __init__(self, db_url_override: str = None):
        self.logger = logging.getLogger("main_system")
        self.agents = {}
        self.is_running = False
        
        # Use the override if provided, otherwise use the default from config
        db_url = db_url_override if db_url_override else config.database.database_url
        
        # Initialize database with the correct URL
        self.db = initialize_database(db_url)
        
    async def initialize(self):
        """Initialize the research system"""
        self.logger.info("Initializing Stock Research System...")
        
        # Validate configuration
        if not config.validate():
            raise RuntimeError("Configuration validation failed")
        
        # Create and start agents
        await self._setup_agents()
        
        self.is_running = True
        self.logger.info("Stock Research System initialized successfully")
    
    async def _setup_agents(self):
        """Setup and start all agents"""
        
        # Create Research Agent
        research_agent = ResearchAgent()
        self.agents["research"] = research_agent
        
        # Create Analysis Agent
        analysis_agent = AnalysisAgent()
        self.agents["analysis"] = analysis_agent
        
        # Create Fact Checker Agent
        fact_checker_agent = FactCheckerAgent()
        self.agents["fact_checker"] = fact_checker_agent
        
        # Connect agents (they can communicate with each other)
        for agent_id, agent in self.agents.items():
            for other_id, other_agent in self.agents.items():
                if agent_id != other_id:
                    agent.add_known_agent(other_id, other_agent)
        
        # Start all agents
        for agent in self.agents.values():
            await agent.start()
        
        self.logger.info(f"Started {len(self.agents)} agents")
    
    async def analyze_stock(self, symbol: str, analysis_type: str = "comprehensive") -> dict:
        """Conduct analysis on a specific stock"""
        self.logger.info(f"Starting {analysis_type} analysis for stock: {symbol}")
        
        if "analysis" not in self.agents:
            raise RuntimeError("Analysis agent not available")
        
        analysis_agent = self.agents["analysis"]
        
        # Create analysis task
        task = Task(
            type=analysis_type,
            data={"symbol": symbol},
            priority=1
        )
        
        # Submit task and wait for completion
        await analysis_agent.add_task(task)
        
        # In a real system, we'd wait for task completion via messaging
        # For now, we'll simulate by waiting a bit
        await asyncio.sleep(3)
        
        self.logger.info(f"Analysis completed for {symbol}")
        return {"status": "completed", "symbol": symbol, "analysis_type": analysis_type}
    
    async def get_data_quality_assessment(self, symbol: str) -> dict:
        """
        A direct method to run 'assess_data_quality' and return the detailed result.
        Perfect for UI integrations.
        """
        self.logger.info(f"Directly running data quality assessment for {symbol}")
        if "fact_checker" not in self.agents:
            raise RuntimeError("Fact Checker agent is not available.")
            
        fact_checker_agent = self.agents["fact_checker"]
        
        # Create a task and process it directly to get the result back
        task = Task(type="assess_data_quality", data={"symbol": symbol})
        result = await fact_checker_agent.process_task(task)
        
        self.logger.info(f"Data quality assessment for {symbol} completed.")
        return result
    
    async def research_stock(self, symbol: str) -> dict:
        """Conduct research on a specific stock"""
        self.logger.info(f"Starting research for stock: {symbol}")
        
        if "research" not in self.agents:
            raise RuntimeError("Research agent not available")
        
        research_agent = self.agents["research"]
        
        # Create research task
        task = Task(
            type="full_research",
            data={"symbol": symbol},
            priority=1
        )
        
        # Submit task and wait for completion
        await research_agent.add_task(task)
        
        # In a real system, we'd wait for task completion via messaging
        # For now, we'll simulate by waiting a bit
        await asyncio.sleep(2)
        
        self.logger.info(f"Research completed for {symbol}")
        return {"status": "completed", "symbol": symbol}
    
    async def research_portfolio(self, symbols: list) -> dict:
        """Research multiple stocks"""
        self.logger.info(f"Starting portfolio research for {len(symbols)} stocks")
        
        results = {}
        for symbol in symbols:
            try:
                result = await self.research_stock(symbol)
                results[symbol] = result
            except Exception as e:
                self.logger.error(f"Error researching {symbol}: {str(e)}")
                results[symbol] = {"status": "failed", "error": str(e)}
        
        return results
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        self.logger.info("Shutting down Stock Research System...")
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
        
        self.is_running = False
        self.logger.info("System shutdown complete")

# async def demo_research():
#     """Demo function to showcase the research system"""
#     system = StockResearchSystem()
    
#     try:
#         # Initialize system
#         await system.initialize()
        
#         # Demo 1: Research a single stock
#         print("\n=== Demo 1: Single Stock Research ===")
#         result = await system.research_stock("RELIANCE.NS")
#         print(f"Research result: {result}")
        
#         # Demo 2: Analyze a stock
#         print("\n=== Demo 2: Stock Analysis ===")
#         analysis_result = await system.analyze_stock("RELIANCE.NS", "technical_analysis")
#         print(f"Technical analysis result: {analysis_result}")
        
#         # Demo 3: Fact-check data quality
#         print("\n=== Demo 3: Data Quality Fact-Check ===")
#         fact_check_result = await system.fact_check_stock("RELIANCE.NS", "assess_data_quality")
#         print(f"Data quality check result: {fact_check_result}")
        
#         # Demo 4: Comprehensive analysis with fact-checking
#         print("\n=== Demo 4: Comprehensive Analysis + Fact-Check ===")
#         comprehensive_result = await system.analyze_stock("TCS.NS", "comprehensive_analysis")
#         print(f"Comprehensive analysis result: {comprehensive_result}")
        
#         # Now fact-check the analysis
#         validation_result = await system.fact_check_stock("TCS.NS", "validate_price_data")
#         print(f"Price data validation result: {validation_result}")
        
#         # Demo 5: Portfolio research and analysis
#         print("\n=== Demo 5: Portfolio Research & Analysis ===")
#         portfolio = ["HDFCBANK.NS", "INFY.NS"]
#         research_results = await system.research_portfolio(portfolio)
#         print(f"Portfolio research results: {json.dumps(research_results, indent=2)}")
        
#         # Demo 6: Individual Agent Tasks
#         print("\n=== Demo 6: Individual Agent Tasks ===")
        
#         # Research Agent tasks
#         research_agent = system.agents["research"]
#         price_task = Task(
#             type="fetch_stock_price",
#             data={"symbol": "TCS.NS"}
#         )
#         await research_agent.add_task(price_task)
        
#         # Analysis Agent tasks
#         analysis_agent = system.agents["analysis"]
        
#         # Technical analysis task
#         tech_task = Task(
#             type="technical_analysis",
#             data={"symbol": "TCS.NS", "period": "3mo"}
#         )
#         await analysis_agent.add_task(tech_task)
        
#         # Risk assessment task
#         risk_task = Task(
#             type="risk_assessment",
#             data={"symbol": "TCS.NS"}
#         )
#         await analysis_agent.add_task(risk_task)
        
#         # Fact Checker Agent tasks
#         fact_checker_agent = system.agents["fact_checker"]
        
#         # Data validation task
#         validation_task = Task(
#             type="validate_price_data",
#             data={"symbol": "TCS.NS", "timeframe": "1mo"}
#         )
#         await fact_checker_agent.add_task(validation_task)
        
#         # Cross-validation task
#         cross_val_task = Task(
#             type="cross_validate_sources",
#             data={"symbol": "TCS.NS", "data_types": ["price", "volume"]}
#         )
#         await fact_checker_agent.add_task(cross_val_task)
        
#         # Consistency check task
#         consistency_task = Task(
#             type="check_data_consistency",
#             data={"symbol": "TCS.NS", "period": "1mo"}
#         )
#         await fact_checker_agent.add_task(consistency_task)
        
#         # Wait for tasks to process
#         await asyncio.sleep(10)
#         print("Individual tasks submitted successfully")
        
#         # Demo 7: Check database content
#         print("\n=== Demo 7: Database Content ===")
#         if stock_operations:
#             try:
#                 stocks = stock_operations.get_all_active_stocks()
#                 print(f"Active stocks in database: {len(stocks)}")
#                 for stock in stocks[:5]:  # Show first 5
#                     print(f"  - {stock.symbol}: {stock.company_name}")
                
#                 # Check latest reports
#                 reports = stock_operations.get_latest_reports(5)
#                 print(f"Latest research reports: {len(reports)}")
#                 for report in reports:
#                     print(f"  - {report.title} ({report.stock.symbol}) - {report.recommendation}")
#             except Exception as e:
#                 print(f"Error accessing database content: {str(e)}")
#         else:
#             print("Stock operations not available")
        
#         # Demo 8: Agent capabilities overview
#         print("\n=== Demo 8: Agent Capabilities ===")
#         for agent_name, agent in system.agents.items():
#             print(f"\n{agent_name.upper()} Agent Capabilities:")
#             capabilities = agent.get_capabilities()
#             for i, capability in enumerate(capabilities, 1):
#                 print(f"  {i}. {capability}")
        
#         # Demo 9: Multi-Agent Workflow
#         print("\n=== Demo 9: Multi-Agent Workflow ===")
#         print("Demonstrating how agents work together:")
#         print("1. Research Agent collects data")
#         print("2. Analysis Agent processes data")
#         print("3. Fact Checker validates results")
        
#         workflow_symbol = "HINDUNILVR.NS"
        
#         # Step 1: Research
#         print(f"\nStep 1: Researching {workflow_symbol}")
#         await system.research_stock(workflow_symbol)
        
#         # Step 2: Analysis
#         print(f"Step 2: Analyzing {workflow_symbol}")
#         await system.analyze_stock(workflow_symbol, "technical_analysis")
        
#         # Step 3: Fact-check
#         print(f"Step 3: Fact-checking {workflow_symbol}")
#         await system.fact_check_stock(workflow_symbol, "assess_data_quality")
        
#         print("Multi-agent workflow completed successfully!")
        
#     except Exception as e:
#         print(f"Demo error: {str(e)}")
#         logging.error(f"Demo error: {str(e)}")
    
#     finally:
#         # Cleanup
#         await system.shutdown()

# async def simple_test():
#     """Simple test to verify basic functionality"""
#     print("=== Simple System Test ===")
    
#     # Test configuration
#     print(f"Configuration loaded: {config.validate()}")
#     print(f"Database URL: {config.database.database_url}")
#     print(f"Stock symbols to track: {config.get_stock_symbols()}")
#     print(f"Claude API available: {'Yes' if config.api.anthropic_api_key else 'No'}")
    
#     # Test database
#     print("\nTesting database...")
#     if stock_operations:
#         stocks = stock_operations.get_all_active_stocks()
#         print(f"Found {len(stocks)} stocks in database")
    
#     # Test agent creation
#     print("\nTesting agent creation...")
    
#     # Research Agent
#     research_agent = ResearchAgent()
#     print(f"Research agent created: {research_agent}")
#     print(f"Research agent capabilities: {research_agent.get_capabilities()}")
    
#     await research_agent.start()
#     print("Research agent started successfully")
#     await research_agent.stop()
#     print("Research agent stopped successfully")
    
#     # Analysis Agent
#     analysis_agent = AnalysisAgent()
#     print(f"Analysis agent created: {analysis_agent}")
#     print(f"Analysis agent capabilities: {analysis_agent.get_capabilities()}")
    
#     await analysis_agent.start()
#     print("Analysis agent started successfully")
#     await analysis_agent.stop()
#     print("Analysis agent stopped successfully")
    
#     # Fact Checker Agent
#     fact_checker_agent = FactCheckerAgent()
#     print(f"Fact checker agent created: {fact_checker_agent}")
#     print(f"Fact checker agent capabilities: {fact_checker_agent.get_capabilities()}")
    
#     await fact_checker_agent.start()
#     print("Fact checker agent started successfully")
#     await fact_checker_agent.stop()
#     print("Fact checker agent stopped successfully")
    
#     print("\n✅ All basic tests passed!")
#     print("\nReady for full demo! Run: python main.py")

async def run_full_system():
    """Initializes the system, researches, and analyzes all active stocks."""
    system = StockResearchSystem()
    
    try:
        await system.initialize()
        all_symbols = config.get_stock_symbols()
        
        # Step 1: Research all stocks to gather data
        print(f"\n=== Starting Research for all {len(all_symbols)} stocks ===")
        await system.research_portfolio(all_symbols)
        print("\n=== Portfolio Research Finished ===")
        
        # ⭐️ FIX: Analyze all stocks to generate reports
        print(f"\n=== Starting Analysis for all {len(all_symbols)} stocks ===")
        for symbol in all_symbols:
            try:
                # This will trigger the AnalysisAgent to create and save a report
                await system.analyze_stock(symbol, "comprehensive_analysis")
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        print("\n=== Portfolio Analysis Finished ===")

    except Exception as e:
        print(f"An error occurred during the full system run: {str(e)}")
        logging.error(f"System run error: {str(e)}", exc_info=True)
    
    finally:
        await system.shutdown()

def main():
    """Main entry point"""
    print("🚀 Stock Research Multi-Agent System")
    print("=" * 50)
    
    # To run the full research on all stocks:
    asyncio.run(run_full_system())

if __name__ == "__main__":
    main()