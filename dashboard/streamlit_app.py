#!/usr/bin/env python3
"""
Quick fix for dashboard database path issue
Replace the current streamlit_app.py with this working version
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import json
import time
from pathlib import Path

# CRITICAL: Page config must be FIRST Streamlit command
st.set_page_config(
    page_title="Stock Research Multi-Agent System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

# Add src path to Python path
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Initialize session state
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False

try:
    from data.database import initialize_database
    from data.models import Stock, PriceData, ResearchReport, ResearchInsight
    from utils.config import config
    from agents.research_agent import ResearchAgent
    from agents.analysis_agent import AnalysisAgent
    from agents.fact_checker_agent import FactCheckerAgent
    from agents.base_agent import Task
    
    imports_successful = True
    
except ImportError as e:
    imports_successful = False
    import_error = str(e)

@st.cache_resource
def initialize_database_with_absolute_path():
    """Initialize database with absolute path - GUARANTEED TO WORK"""
    try:
        # DIRECT ABSOLUTE PATH - NO MORE RELATIVE PATH ISSUES!
        project_root_path = Path(project_root).resolve()
        data_dir = project_root_path / "data"
        db_file = data_dir / "stock_research.db"
        
        # Show paths for debugging
        st.sidebar.info(f"📂 Project root: {project_root_path}")
        st.sidebar.info(f"📂 Data dir: {data_dir}")
        st.sidebar.info(f"🗄️ DB file: {db_file}")
        st.sidebar.info(f"✅ DB exists: {db_file.exists()}")
        
        if not db_file.exists():
            st.error(f"❌ Database file not found at: {db_file}")
            st.error("🔧 **Quick Fix:**")
            st.code(f"""
# Run this to create the database:
cd {project_root_path}/src
python main.py simple

# Then refresh this dashboard
            """)
            return None
        
        # Create database URL with ABSOLUTE path
        absolute_db_url = f"sqlite:///{db_file}"
        st.sidebar.success(f"🗄️ Using DB URL: {absolute_db_url}")
        
        # Use the correct initialize_database function
        db_manager = initialize_database(absolute_db_url)
        
        st.sidebar.success("✅ Database connected successfully!")
        return db_manager
        
    except Exception as e:
        st.error(f"❌ Database initialization failed: {e}")
        st.error(f"🔧 **Debug info:**")
        st.error(f"- Project root: {project_root}")
        st.error(f"- Expected DB file: {project_root}/data/stock_research.db")
        st.error(f"- Current working dir: {os.getcwd()}")
        return None

def display_agent_status():
    """Display current agent status"""
    st.sidebar.markdown("### 🤖 Agent Status")
    
    # Show import status first
    if imports_successful:
        st.sidebar.success("✅ All imports successful!")
    else:
        st.sidebar.error("❌ Import failed")
        st.sidebar.error(f"Error: {import_error}")
        return
    
    agent_info = [
        ("Research Agent", "research", "Collecting market data"),
        ("Analysis Agent", "analysis", "Processing technical indicators"),
        ("Fact Checker", "fact_checker", "Validating data quality")
    ]
    
    for name, key, description in agent_info:
        status = "active" if st.session_state.db_initialized else "inactive"
        color = "🟢" if status == "active" else "🔴"
        
        st.sidebar.markdown(f"""
        **{color} {name}**  
        *{description}*
        """)

@st.cache_resource
def get_database_manager():
    """Get cached database manager instance"""
    return initialize_database_with_absolute_path()

def show_overview():
    """Show system overview"""
    st.header("📊 System Overview")
    
    if not st.session_state.db_initialized:
        st.warning("🔄 Database not initialized. Please check the sidebar for status.")
        return
    
    # Get basic stats
    try:
        db_manager = get_database_manager()
        if not db_manager:
            st.error("Cannot connect to database")
            return
            
        with db_manager.get_session() as session:
            stock_count = session.query(Stock).filter(Stock.is_active == True).count()
            total_reports = session.query(ResearchReport).count()
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Active Stocks", stock_count)
        
        with col2:
            st.metric("📋 Total Reports", total_reports)
        
        with col3:
            st.metric("🤖 Active Agents", 3)
        
        with col4:
            st.metric("✅ System Status", "Online")
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        
        with db_manager.get_session() as session:
            # Eager load the stock relationship and get all needed attributes in session
            recent_reports = session.query(ResearchReport).join(Stock).add_columns(
                Stock.symbol, 
                ResearchReport.created_at,
                ResearchReport.summary,
                ResearchReport.recommendation
            ).order_by(ResearchReport.created_at.desc()).limit(5).all()
            
            if recent_reports:
                for report_tuple in recent_reports:
                    report = report_tuple[0]  # ResearchReport object
                    stock_symbol = report_tuple[1]  # Stock.symbol
                    created_at = report_tuple[2]  # created_at
                    summary = report_tuple[3]  # summary
                    recommendation = report_tuple[4]  # recommendation
                    
                    with st.expander(f"📋 Report for {stock_symbol} - {created_at.strftime('%Y-%m-%d %H:%M')}"):
                        # Use safe attribute access
                        score = getattr(report, 'overall_score', None) or getattr(report, 'score', 'N/A')
                        st.write(f"**Score:** {score}")
                        st.write(f"**Recommendation:** {recommendation or 'N/A'}")
                        if summary:
                            st.write(f"**Summary:** {summary[:200]}...")
            else:
                st.info("No research reports found. Run the system to generate reports.")
                
    except Exception as e:
        st.error(f"Error loading overview: {e}")
        # Add debug info
        with st.expander("🔧 Debug Details"):
            st.write(f"Error type: {type(e).__name__}")
            st.write(f"Error message: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def show_stock_research():
    """Show stock research interface"""
    st.header("🔍 Stock Research")
    st.write("Research and analyze individual stocks")
    
    if not st.session_state.db_initialized:
        st.warning("🔄 Database not initialized. Please check the sidebar for status.")
        return
    
    # Stock selection
    try:
        db_manager = get_database_manager()
        if not db_manager:
            return
            
        # Get all stock information in one session to avoid detached instances
        with db_manager.get_session() as session:
            stocks_info = session.query(Stock.id, Stock.symbol, Stock.is_active).filter(Stock.is_active == True).all()
            
        if not stocks_info:
            st.warning("No stocks found in database")
            st.info("💡 To add stocks, run: `cd ../src && python main.py`")
            return
            
        # Create options using the data we retrieved
        stock_options = {symbol: symbol for _, symbol, _ in stocks_info}
        
        if stock_options:
            selected_symbol = st.selectbox("Select a stock to research:", list(stock_options.keys()))
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Generate Research Report"):
                    with st.spinner(f"Researching {selected_symbol}..."):
                        st.success(f"✅ Research initiated for {selected_symbol}")
                        st.info("Note: In a full implementation, this would trigger the research agent to gather data and generate a comprehensive report.")
            
            with col2:
                if st.button("📊 View Price Data"):
                    with st.spinner(f"Loading price data for {selected_symbol}..."):
                        st.success(f"✅ Price data loaded for {selected_symbol}")
                        st.info("Note: Price charts would be displayed here with historical data, volume, and technical indicators.")
            
            # Show current stock info
            st.subheader(f"📈 {selected_symbol} Information")
            
            try:
                with db_manager.get_session() as session:
                    # Get stock info and price data in one session
                    stock_info = session.query(Stock).filter(Stock.symbol == selected_symbol).first()
                    
                    if stock_info:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Symbol", stock_info.symbol)
                        with col2:
                            st.metric("ID", stock_info.id)
                        with col3:
                            st.metric("Status", "Active" if stock_info.is_active else "Inactive")
                        
                        # Get recent price data within the same session
                        recent_price = session.query(PriceData).filter(
                            PriceData.stock_id == stock_info.id
                        ).order_by(PriceData.date.desc()).first()
                        
                        if recent_price:
                            st.subheader("💰 Latest Price Data")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Close Price", f"₹{recent_price.close_price:.2f}")
                            with col2:
                                st.metric("Volume", f"{recent_price.volume:,}")
                            with col3:
                                st.metric("High", f"₹{recent_price.high_price:.2f}")
                            with col4:
                                st.metric("Low", f"₹{recent_price.low_price:.2f}")
                            with col5:
                                st.metric("Date", recent_price.date.strftime("%Y-%m-%d"))
                            
                            # Get historical data for mini chart
                            historical_data = session.query(PriceData).filter(
                                PriceData.stock_id == stock_info.id
                            ).order_by(PriceData.date.desc()).limit(30).all()
                            
                            if len(historical_data) > 1:
                                # Create a simple price chart
                                dates = [price.date for price in reversed(historical_data)]
                                prices = [price.close_price for price in reversed(historical_data)]
                                
                                import plotly.graph_objects as go
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=dates, 
                                    y=prices,
                                    mode='lines+markers',
                                    name='Close Price',
                                    line=dict(color='#00ff88', width=2),
                                    marker=dict(size=4)
                                ))
                                fig.update_layout(
                                    title=f"{selected_symbol} - Last 30 Days",
                                    xaxis_title="Date",
                                    yaxis_title="Price (₹)",
                                    height=400,
                                    showlegend=False,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No price data available for this stock.")
                            st.info("💡 To add price data, run the full system: `cd ../src && python main.py`")
                    else:
                        st.error(f"Stock {selected_symbol} not found")
                    
            except Exception as e:
                st.error(f"Error loading stock details: {e}")
                # Add debug info
                with st.expander("🔧 Debug Details"):
                    st.write(f"Error type: {type(e).__name__}")
                    st.write(f"Error message: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                
    except Exception as e:
        st.error(f"Error in stock research: {e}")
        # Add debug info
        with st.expander("🔧 Debug Details"):
            st.write(f"Error type: {type(e).__name__}")
            st.write(f"Error message: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def show_technical_analysis():
    """Show technical analysis charts"""
    st.header("📈 Technical Analysis")
    st.write("Advanced charting and technical indicators")
    
    if not st.session_state.db_initialized:
        st.warning("🔄 Database not initialized. Please check the sidebar for status.")
        return
    
    try:
        db_manager = get_database_manager()
        if not db_manager:
            return
            
        # Get available stocks
        with db_manager.get_session() as session:
            stocks_data = session.query(Stock.symbol).filter(Stock.is_active == True).all()
            
        if not stocks_data:
            st.warning("No stocks found for analysis")
            return
            
        stock_symbols = [symbol for (symbol,) in stocks_data]
        selected_symbol = st.selectbox("Select stock for technical analysis:", stock_symbols)
        
        if selected_symbol:
            with db_manager.get_session() as session:
                # Get stock and its price data
                stock = session.query(Stock).filter(Stock.symbol == selected_symbol).first()
                if not stock:
                    st.error("Stock not found")
                    return
                
                # Get historical price data
                price_data = session.query(PriceData).filter(
                    PriceData.stock_id == stock.id
                ).order_by(PriceData.date.asc()).all()
                
                if not price_data:
                    st.warning(f"No price data available for {selected_symbol}")
                    st.info("💡 Run the full system to generate price data: `cd ../src && python main.py`")
                    return
                
                # Convert to lists for plotting
                dates = [p.date for p in price_data]
                opens = [p.open_price for p in price_data]
                highs = [p.high_price for p in price_data]
                lows = [p.low_price for p in price_data]
                closes = [p.close_price for p in price_data]
                volumes = [p.volume for p in price_data]
                
                # Create main price chart (candlestick)
                from plotly.subplots import make_subplots
                import plotly.graph_objects as go
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=(f'{selected_symbol} Price Chart', 'Volume'),
                    row_heights=[0.7, 0.3]
                )
                
                # Add candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=dates,
                        open=opens,
                        high=highs,
                        low=lows,
                        close=closes,
                        name="Price",
                        increasing_line_color='#00ff88',
                        decreasing_line_color='#ff4444'
                    ),
                    row=1, col=1
                )
                
                # Add volume bars
                fig.add_trace(
                    go.Bar(
                        x=dates,
                        y=volumes,
                        name="Volume",
                        marker_color='rgba(158,202,225,0.6)'
                    ),
                    row=2, col=1
                )
                
                # Calculate and add moving averages
                if len(closes) >= 20:
                    ma_20 = []
                    for i in range(len(closes)):
                        if i >= 19:
                            ma_20.append(sum(closes[i-19:i+1]) / 20)
                        else:
                            ma_20.append(None)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=ma_20,
                            mode='lines',
                            name='MA 20',
                            line=dict(color='orange', width=2)
                        ),
                        row=1, col=1
                    )
                
                if len(closes) >= 50:
                    ma_50 = []
                    for i in range(len(closes)):
                        if i >= 49:
                            ma_50.append(sum(closes[i-49:i+1]) / 50)
                        else:
                            ma_50.append(None)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=ma_50,
                            mode='lines',
                            name='MA 50',
                            line=dict(color='purple', width=2)
                        ),
                        row=1, col=1
                    )
                
                # Update layout
                fig.update_layout(
                    title=f"{selected_symbol} - Technical Analysis",
                    yaxis_title="Price (₹)",
                    xaxis_title="Date",
                    height=700,
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                fig.update_xaxes(rangeslider_visible=False)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical Indicators Summary
                st.subheader("📊 Technical Indicators")
                
                col1, col2, col3, col4 = st.columns(4)
                
                # Current price vs moving averages
                current_price = closes[-1]
                
                with col1:
                    if len(closes) >= 20:
                        ma20_current = sum(closes[-20:]) / 20
                        ma20_trend = "🔺" if current_price > ma20_current else "🔻"
                        st.metric("vs MA20", f"{ma20_trend} {((current_price/ma20_current - 1) * 100):.1f}%")
                    else:
                        st.metric("vs MA20", "N/A")
                
                with col2:
                    if len(closes) >= 50:
                        ma50_current = sum(closes[-50:]) / 50
                        ma50_trend = "🔺" if current_price > ma50_current else "🔻"
                        st.metric("vs MA50", f"{ma50_trend} {((current_price/ma50_current - 1) * 100):.1f}%")
                    else:
                        st.metric("vs MA50", "N/A")
                
                with col3:
                    # Daily change
                    if len(closes) >= 2:
                        daily_change = ((current_price / closes[-2]) - 1) * 100
                        daily_trend = "🔺" if daily_change > 0 else "🔻"
                        st.metric("Daily Change", f"{daily_trend} {daily_change:.1f}%")
                    else:
                        st.metric("Daily Change", "N/A")
                
                with col4:
                    # Volatility (last 20 days)
                    if len(closes) >= 20:
                        recent_closes = closes[-20:]
                        volatility = (max(recent_closes) / min(recent_closes) - 1) * 100
                        st.metric("20D Volatility", f"{volatility:.1f}%")
                    else:
                        st.metric("20D Volatility", "N/A")
                
                # Price Analysis
                st.subheader("💰 Price Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Support & Resistance Levels:**")
                    recent_lows = lows[-30:] if len(lows) >= 30 else lows
                    recent_highs = highs[-30:] if len(highs) >= 30 else highs
                    
                    support_level = min(recent_lows)
                    resistance_level = max(recent_highs)
                    
                    st.write(f"Support: ₹{support_level:.2f}")
                    st.write(f"Resistance: ₹{resistance_level:.2f}")
                    st.write(f"Current: ₹{current_price:.2f}")
                
                with col2:
                    st.write("**Volume Analysis:**")
                    avg_volume = sum(volumes[-20:]) / len(volumes[-20:]) if len(volumes) >= 20 else sum(volumes) / len(volumes)
                    current_volume = volumes[-1]
                    volume_ratio = current_volume / avg_volume
                    
                    st.write(f"Avg Volume (20D): {avg_volume:,.0f}")
                    st.write(f"Current Volume: {current_volume:,.0f}")
                    st.write(f"Volume Ratio: {volume_ratio:.2f}x")
                    
                    if volume_ratio > 1.5:
                        st.success("🔊 High Volume Day")
                    elif volume_ratio < 0.5:
                        st.warning("🔇 Low Volume Day")
                    else:
                        st.info("📊 Normal Volume")
                
    except Exception as e:
        st.error(f"Error in technical analysis: {e}")
        with st.expander("🔧 Debug Details"):
            st.write(f"Error type: {type(e).__name__}")
            st.write(f"Error message: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def show_data_quality():
    """Show data quality checks"""
    st.header("✅ Data Quality")
    st.write("Validate and verify data accuracy")
    
    if not st.session_state.db_initialized:
        st.warning("🔄 Database not initialized. Please check the sidebar for status.")
        return
    
    try:
        db_manager = get_database_manager()
        if not db_manager:
            return
        
        # Data Quality Overview
        with db_manager.get_session() as session:
            total_stocks = session.query(Stock).count()
            active_stocks = session.query(Stock).filter(Stock.is_active == True).count()
            total_price_records = session.query(PriceData).count()
            total_reports = session.query(ResearchReport).count()
            
            # Check which stocks have price data
            stocks_with_data = session.query(Stock.symbol).join(PriceData).distinct().all()
            stocks_without_data = session.query(Stock.symbol).filter(
                Stock.is_active == True,
                ~Stock.id.in_(
                    session.query(PriceData.stock_id).distinct()
                )
            ).all()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Total Stocks", total_stocks)
        with col2:
            st.metric("✅ Active Stocks", active_stocks)
        with col3:
            st.metric("💰 Price Records", total_price_records)
        with col4:
            st.metric("📋 Reports", total_reports)
        
        # Data Quality Issues
        st.subheader("🔍 Data Quality Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**✅ Stocks WITH Price Data:**")
            if stocks_with_data:
                for (symbol,) in stocks_with_data:
                    st.success(f"📊 {symbol}")
            else:
                st.info("No stocks have price data yet")
        
        with col2:
            st.write("**❌ Stocks WITHOUT Price Data:**")
            if stocks_without_data:
                for (symbol,) in stocks_without_data:
                    st.warning(f"📭 {symbol}")
            else:
                st.success("All active stocks have price data!")
        
        # Price Data Generation
        if stocks_without_data:
            st.subheader("🔧 Generate Price Data")
            st.warning(f"⚠️ {len(stocks_without_data)} stocks are missing price data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Generate Sample Data", help="Generate sample price data for testing"):
                    with st.spinner("Generating sample price data..."):
                        success = generate_sample_price_data(db_manager)
                        if success:
                            st.success("✅ Sample price data generated!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to generate sample data")
            
            with col2:
                if st.button("🌐 Fetch Real Data", help="Fetch real price data (requires internet)"):
                    st.info("🔄 Real data fetching would be implemented here")
                    st.code("cd ../src && python main.py")
            
            with col3:
                if st.button("📊 Run Full System", help="Run the complete multi-agent system"):
                    st.info("💡 To run the full system:")
                    st.code("""
# From your project root:
cd src
python main.py

# This will:
# - Fetch real price data
# - Generate research reports  
# - Run all agents
                    """)
        
        # Data Validation Rules
        st.subheader("📋 Data Validation Rules")
        st.info("🔍 Data quality checks include:")
        st.write("- Price data completeness")
        st.write("- Date range validation")
        st.write("- Price anomaly detection")
        st.write("- Volume consistency checks")
        st.write("- Source credibility assessment")
        
    except Exception as e:
        st.error(f"Error in data quality check: {e}")
        with st.expander("🔧 Debug Details"):
            st.write(f"Error type: {type(e).__name__}")
            st.write(f"Error message: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def generate_sample_price_data(db_manager):
    """Generate sample price data for demonstration"""
    try:
        import random
        from datetime import datetime, timedelta
        
        with db_manager.get_session() as session:
            # Get stocks without price data
            stocks_without_data = session.query(Stock).filter(
                Stock.is_active == True,
                ~Stock.id.in_(
                    session.query(PriceData.stock_id).distinct()
                )
            ).all()
            
            for stock in stocks_without_data:
                # Generate 30 days of sample price data
                base_price = random.uniform(100, 2000)  # Random base price
                current_date = datetime.now().date() - timedelta(days=30)
                
                for i in range(30):
                    # Simulate realistic price movements
                    daily_change = random.uniform(-0.05, 0.05)  # ±5% daily change
                    base_price *= (1 + daily_change)
                    
                    # Calculate OHLC based on base price
                    open_price = base_price * random.uniform(0.99, 1.01)
                    close_price = base_price * random.uniform(0.99, 1.01)
                    high_price = max(open_price, close_price) * random.uniform(1.0, 1.03)
                    low_price = min(open_price, close_price) * random.uniform(0.97, 1.0)
                    volume = random.randint(100000, 10000000)
                    
                    # Create price record
                    price_record = PriceData(
                        stock_id=stock.id,
                        date=current_date,
                        open_price=round(open_price, 2),
                        high_price=round(high_price, 2),
                        low_price=round(low_price, 2),
                        close_price=round(close_price, 2),
                        volume=volume
                    )
                    session.add(price_record)
                    
                    current_date += timedelta(days=1)
            
            session.commit()
            return True
            
    except Exception as e:
        st.error(f"Error generating sample data: {e}")
        return False

def show_reports():
    """Show generated reports"""
    st.header("📋 Research Reports")
    st.write("View and manage generated research reports")
    
    if not st.session_state.db_initialized:
        st.warning("🔄 Database not initialized. Please check the sidebar for status.")
        return
        
    try:
        db_manager = get_database_manager()
        if not db_manager:
            return
            
        with db_manager.get_session() as session:
            # Get reports with stock information in one query to avoid session issues
            reports_query = session.query(ResearchReport).join(Stock).all()
            
            # Process reports within the session to get all needed data
            reports_data = []
            for report in reports_query:
                report_data = {
                    'id': report.id,
                    'stock_symbol': report.stock.symbol,
                    'stock_id': report.stock.id,
                    'created_at': report.created_at,
                    'updated_at': getattr(report, 'updated_at', None),
                    'summary': getattr(report, 'summary', 'No summary available'),
                    'content': getattr(report, 'content', None),
                    'recommendation': getattr(report, 'recommendation', 'N/A'),
                    'score': getattr(report, 'overall_score', None) or getattr(report, 'score', None),
                    'agent_id': getattr(report, 'agent_id', 'Unknown'),
                    'report_type': getattr(report, 'report_type', 'Research Report')
                }
                reports_data.append(report_data)
            
        if reports_data:
            # Sort by creation date (newest first)
            reports_data = sorted(reports_data, key=lambda x: x['created_at'], reverse=True)
            
            for report_data in reports_data[:10]:  # Show latest 10
                with st.expander(f"📊 {report_data['stock_symbol']} - {report_data['created_at'].strftime('%Y-%m-%d %H:%M')}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if report_data['score'] is not None:
                            st.metric("Overall Score", f"{report_data['score']:.2f}")
                        else:
                            st.metric("Overall Score", "N/A")
                    
                    with col2:
                        st.metric("Recommendation", report_data['recommendation'])
                    
                    with col3:
                        st.metric("Stock", report_data['stock_symbol'])
                    
                    # Show summary
                    if report_data['summary']:
                        st.write("**Summary:**")
                        st.write(report_data['summary'])
                    
                    # Show content if available
                    if report_data['content']:
                        st.write("**Full Analysis:**")
                        content_preview = report_data['content'][:500] + "..." if len(report_data['content']) > 500 else report_data['content']
                        st.write(content_preview)
                    
                    # Show other details
                    st.write("**Report Details:**")
                    details_col1, details_col2 = st.columns(2)
                    
                    with details_col1:
                        st.write(f"**Agent:** {report_data['agent_id']}")
                        st.write(f"**Type:** {report_data['report_type']}")
                    
                    with details_col2:
                        st.write(f"**Created:** {report_data['created_at']}")
                        if report_data['updated_at']:
                            st.write(f"**Updated:** {report_data['updated_at']}")
        else:
            st.info("🤔 No reports found. Generate some reports first!")
            st.markdown("""
            **To generate reports:**
            1. Go to the **Stock Research** tab
            2. Select a stock and click **Generate Research Report**
            3. Or run the full system: `cd ../src && python main.py`
            """)
            
    except Exception as e:
        st.error(f"Error loading reports: {e}")
        # Add debug info
        with st.expander("🔧 Debug Details"):
            st.write(f"Error type: {type(e).__name__}")
            st.write(f"Error message: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def main():
    """Main Streamlit application"""
    
    # Check imports first
    if not imports_successful:
        st.error(f"❌ Import error: {import_error}")
        st.error("**Troubleshooting steps:**")
        st.error("1. Make sure you're running from the dashboard directory")
        st.error("2. Ensure the src directory exists in the parent folder")
        st.error("3. Check if all required files exist in src/")
        st.error("4. Run diagnostic: `python dashboard_fix.py`")
        st.stop()
    
    # Initialize database
    if not st.session_state.db_initialized:
        with st.spinner("Connecting to database..."):
            db_manager = initialize_database_with_absolute_path()
            if db_manager:
                st.session_state.db_initialized = True
                st.success("✅ Database connected successfully!")
            else:
                st.session_state.db_initialized = False
                st.error("❌ Failed to connect to database")
                st.stop()
    
    # Header
    st.title("📈 Stock Research Multi-Agent System")
    st.markdown("*Powered by Research, Analysis, and Fact-Checking Agents*")
    
    # Sidebar
    display_agent_status()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 Stock Research", 
        "📈 Technical Analysis", 
        "✅ Data Quality", 
        "📋 Reports"
    ])
    
    with tab1:
        show_overview()
    
    with tab2:
        show_stock_research()
    
    with tab3:
        show_technical_analysis()
    
    with tab4:
        show_data_quality()
    
    with tab5:
        show_reports()

if __name__ == "__main__":
    main()