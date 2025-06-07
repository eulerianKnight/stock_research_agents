import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from typing import Generator
import logging
from datetime import datetime

from .models import (
    Base,
    Stock,
    PriceData,
    FinancialData,
    ResearchReport,
    ResearchInsight,
    AgentTask,
    MarketNews,
)


class DatabaseManager:
    """Manages database connections and operations"""

    def __init__(self, database_url: str):
        if database_url is None:
            # Default to local SQLite database
            db_path = os.path.join(os.getcwd(), "data", "stock_research.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            database_url = f"sqlite:///{db_path}"

        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        self.logger = logging.getLogger("database")

    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating tables: {str(e)}")
            raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()

    def get_session_direct(self) -> Session:
        """Get a session without context manager (manual cleanup required)"""
        return self.SessionLocal()

    def initialize_sample_data(self):
        """Initialize the database with sample Indian stocks"""
        sample_stocks = [
            {
                "symbol": "RELIANCE.NS",
                "company_name": "Reliance Industries Limited",
                "sector": "Energy",
                "industry": "Oil & Gas Refining & Marketing",
                "exchange": "NSE",
                "isin": "INE002A01018",
            },
            {
                "symbol": "TCS.NS",
                "company_name": "Tata Consultancy Services Limited",
                "sector": "Information Technology",
                "industry": "IT Services & Consulting",
                "exchange": "NSE",
                "isin": "INE467B01029",
            },
            {
                "symbol": "HDFCBANK.NS",
                "company_name": "HDFC Bank Limited",
                "sector": "Financial Services",
                "industry": "Private Sector Bank",
                "exchange": "NSE",
                "isin": "INE040A01034",
            },
            {
                "symbol": "INFY.NS",
                "company_name": "Infosys Limited",
                "sector": "Information Technology",
                "industry": "IT Services & Consulting",
                "exchange": "NSE",
                "isin": "INE009A01021",
            },
            {
                "symbol": "HINDUNILVR.NS",
                "company_name": "Hindustan Unilever Limited",
                "sector": "Fast Moving Consumer Goods",
                "industry": "Personal Care",
                "exchange": "NSE",
                "isin": "INE030A01027",
            },
            {
                "symbol": "ICICIBANK.NS",
                "company_name": "ICICI Bank Limited",
                "sector": "Financial Services",
                "industry": "Private Sector Bank",
                "exchange": "NSE",
                "isin": "INE090A01021",
            },
            {
                "symbol": "KOTAKBANK.NS",
                "company_name": "Kotak Mahindra Bank Limited",
                "sector": "Financial Services",
                "industry": "Private Sector Bank",
                "exchange": "NSE",
                "isin": "INE237A01028",
            },
            {
                "symbol": "LT.NS",
                "company_name": "Larsen & Toubro Limited",
                "sector": "Capital Goods",
                "industry": "Construction & Engineering",
                "exchange": "NSE",
                "isin": "INE018A01030",
            },
            {
                "symbol": "SBIN.NS",
                "company_name": "State Bank of India",
                "sector": "Financial Services",
                "industry": "Public Sector Bank",
                "exchange": "NSE",
                "isin": "INE062A01020",
            },
            {
                "symbol": "BHARTIARTL.NS",
                "company_name": "Bharti Airtel Limited",
                "sector": "Telecommunication",
                "industry": "Telecom Services",
                "exchange": "NSE",
                "isin": "INE397D01024",
            },
        ]

        try:
            with self.get_session() as session:
                # Check if data already exists
                existing_count = session.query(Stock).count()
                if existing_count > 0:
                    self.logger.info(
                        f"Sample data already exists ({existing_count} stocks)"
                    )
                    return

                # Add sample stocks
                for stock_data in sample_stocks:
                    stock = Stock(**stock_data)
                    session.add(stock)

                session.commit()
                self.logger.info(
                    f"Added {len(sample_stocks)} sample stocks to database"
                )

        except SQLAlchemyError as e:
            self.logger.error(f"Error initializing sample data: {str(e)}")
            raise


# Database operations helper class
class StockDataOperations:
    """Helper class for common stock data operations"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger("stock_operations")

    def get_stock_by_symbol(self, symbol: str) -> Stock:
        """Get stock by symbol"""
        with self.db.get_session() as session:
            return session.query(Stock).filter(Stock.symbol == symbol).first()

    def get_all_active_stocks(self):
        """Get all active stocks"""
        with self.db.get_session() as session:
            return session.query(Stock).filter(Stock.is_active == True).all()

    def save_price_data(self, stock_symbol: str, price_data_list):
        """Save price data for a stock"""
        with self.db.get_session() as session:
            stock = session.query(Stock).filter(Stock.symbol == stock_symbol).first()
            if not stock:
                self.logger.error(f"Stock {stock_symbol} not found")
                return False

            for price_data in price_data_list:
                existing = (
                    session.query(PriceData)
                    .filter(
                        PriceData.stock_id == stock.id,
                        PriceData.date == price_data.get("date"),
                    )
                    .first()
                )

                if not existing:
                    price_record = PriceData(stock_id=stock.id, **price_data)
                    session.add(price_record)

            return True
        
    def save_financial_data(self, stock_symbol: str, financial_data_dict: dict):
        """Save financial data for a stock."""
        try:
            with self.db.get_session() as session:
                stock = session.query(Stock).filter(Stock.symbol == stock_symbol).first()
                if not stock:
                    self.logger.error(f"Stock {stock_symbol} not found, cannot save financial data.")
                    return False

                current_year = datetime.now().year
                
                # Check if a record for this stock and year already exists
                existing = session.query(FinancialData).filter_by(stock_id=stock.id, year=current_year).first()

                if not existing:
                    record = FinancialData(
                        stock_id=stock.id,
                        period="Annual",
                        year=current_year,
                        revenue=financial_data_dict.get("totalRevenue"),
                        net_income=financial_data_dict.get("netIncomeToCommon"),
                        eps=financial_data_dict.get("trailingEps"),
                        pe_ratio=financial_data_dict.get("trailingPE"),
                        pb_ratio=financial_data_dict.get("priceToBook"),
                        debt_to_equity=financial_data_dict.get("debtToEquity"),
                        roe=financial_data_dict.get("returnOnEquity"),
                        roa=financial_data_dict.get("returnOnAssets"),
                        dividend_yield=financial_data_dict.get("dividendYield"),
                        data_source="yahoo_finance"
                    )
                    session.add(record)
                    self.logger.info(f"SUCCESS: Added new financial data for {stock_symbol} for {current_year}")
                else:
                    self.logger.info(f"INFO: Financial data for {stock_symbol} already exists for {current_year}")
                return True
        except Exception as e:
            self.logger.error(f"CRITICAL ERROR saving financial data for {stock_symbol}: {e}", exc_info=True)
            return False
        
    def save_market_news(self, stock_symbol: str, news_list: list):
        """Save market news for a stock."""
        try:
            with self.db.get_session() as session:
                stock = session.query(Stock).filter_by(symbol=stock_symbol).one_or_none()
                if not stock:
                    self.logger.warning(f"Stock {stock_symbol} not found for saving news.")
                    return False

                for news_item in news_list:
                    # Check if news with the same URL already exists
                    existing = session.query(MarketNews).filter_by(source_url=news_item.get("url")).first()
                    if not existing and news_item.get("url"):
                        record = MarketNews(
                            stock_id=stock.id,
                            title=news_item.get("title"),
                            source_url=news_item.get("url"),
                            source=news_item.get("publisher"),
                            summary=news_item.get("summary"),
                            published_at=news_item.get("published_at")
                        )
                        session.add(record)
                self.logger.info(f"Saved {len(news_list)} news items for {stock_symbol}")
                return True
        except Exception as e:
            self.logger.error(f"Error saving market news for {stock_symbol}: {e}", exc_info=True)
            return False

    def save_research_report(self, stock_symbol: str, report_data):
        """Save a research report"""
        with self.db.get_session() as session:
            stock = session.query(Stock).filter(Stock.symbol == stock_symbol).first()
            if not stock:
                self.logger.error(f"Stock {stock_symbol} not found")
                return None

            report = ResearchReport(stock_id=stock.id, **report_data)
            session.add(report)
            session.flush()  # Get the ID

            return report.id

    def get_latest_reports(self, limit: int = 10):
        """Get latest research reports"""
        with self.db.get_session() as session:
            return (
                session.query(ResearchReport)
                .join(Stock)
                .order_by(ResearchReport.created_at.desc())
                .limit(limit)
                .all()
            )


# Global database instance (will be initialized in main.py)
db_manager = None
stock_operations = None


def initialize_database(database_url: str):
    """Initialize the global database manager"""
    global db_manager, stock_operations

    db_manager = DatabaseManager(database_url)
    db_manager.create_tables()
    db_manager.initialize_sample_data()

    stock_operations = StockDataOperations(db_manager)

    return db_manager
