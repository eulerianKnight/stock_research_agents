# src/data/models.py

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional

Base = declarative_base()


class Stock(Base):
    """Stock information table"""

    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    company_name = Column(String(200), nullable=False)
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    exchange = Column(String(10))  # NSE, BSE
    isin = Column(String(12))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    price_data = relationship("PriceData", back_populates="stock")
    research_reports = relationship("ResearchReport", back_populates="stock")
    financial_data = relationship("FinancialData", back_populates="stock")


class PriceData(Base):
    """Historical and current price data"""

    __tablename__ = "price_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer)
    adjusted_close = Column(Float)
    change_percent = Column(Float)
    data_source = Column(String(50))  # yahoo, nse_api, etc.
    created_at = Column(DateTime, default=func.now())

    # Relationships
    stock = relationship("Stock", back_populates="price_data")


class FinancialData(Base):
    """Fundamental financial data"""

    __tablename__ = "financial_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    period = Column(String(10))  # Q1, Q2, Q3, Q4, Annual
    year = Column(Integer, nullable=False)

    # Financial metrics
    revenue = Column(Float)
    net_income = Column(Float)
    eps = Column(Float)  # Earnings per share
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    debt_to_equity = Column(Float)
    roe = Column(Float)  # Return on equity
    roa = Column(Float)  # Return on assets
    dividend_yield = Column(Float)

    # Ratios
    current_ratio = Column(Float)
    quick_ratio = Column(Float)
    operating_margin = Column(Float)
    profit_margin = Column(Float)

    data_source = Column(String(50))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    stock = relationship("Stock", back_populates="financial_data")


class ResearchReport(Base):
    """Generated research reports"""

    __tablename__ = "research_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    report_type = Column(
        String(50), nullable=False
    )  # technical, fundamental, news, combined

    # Report content
    title = Column(String(200), nullable=False)
    summary = Column(Text)
    detailed_analysis = Column(Text)
    recommendation = Column(String(20))  # BUY, SELL, HOLD
    confidence_score = Column(Float)  # 0.0 to 1.0
    price_target = Column(Float)
    risk_level = Column(String(20))  # LOW, MEDIUM, HIGH

    # Agent information
    generated_by_agent = Column(String(100))
    fact_checked = Column(Boolean, default=False)
    fact_check_score = Column(Float)

    # Metadata
    data_sources = Column(Text)  # JSON string of sources used
    analysis_date = Column(DateTime, nullable=False)
    valid_until = Column(DateTime)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    stock = relationship("Stock", back_populates="research_reports")
    insights = relationship("ResearchInsight", back_populates="report")


class ResearchInsight(Base):
    """Individual insights within a research report"""

    __tablename__ = "research_insights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(Integer, ForeignKey("research_reports.id"), nullable=False)

    insight_type = Column(
        String(50), nullable=False
    )  # technical, fundamental, news, sentiment
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    impact_score = Column(Float)  # -1.0 to 1.0 (negative to positive impact)
    confidence_level = Column(Float)  # 0.0 to 1.0

    # Supporting data
    supporting_data = Column(Text)  # JSON string with charts, numbers, etc.
    source_urls = Column(Text)  # JSON array of source URLs

    created_at = Column(DateTime, default=func.now())

    # Relationships
    report = relationship("ResearchReport", back_populates="insights")


class AgentTask(Base):
    """Track agent tasks and their status"""

    __tablename__ = "agent_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(36), unique=True, nullable=False)  # UUID
    agent_id = Column(String(100), nullable=False)
    task_type = Column(String(50), nullable=False)

    # Task details
    input_data = Column(Text)  # JSON string
    output_data = Column(Text)  # JSON string
    status = Column(
        String(20), default="pending"
    )  # pending, processing, completed, failed
    error_message = Column(Text)

    # Timing
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    processing_time_seconds = Column(Float)


class MarketNews(Base):
    """Store market news and events"""

    __tablename__ = "market_news"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(
        Integer, ForeignKey("stocks.id"), nullable=True
    )  # Can be general market news

    title = Column(String(300), nullable=False)
    content = Column(Text)
    summary = Column(Text)
    sentiment_score = Column(Float)  # -1.0 to 1.0
    impact_level = Column(String(20))  # LOW, MEDIUM, HIGH

    # Source information
    source = Column(String(100))
    source_url = Column(String(500))
    published_at = Column(DateTime, nullable=False)

    # Processing information
    processed_by_agent = Column(String(100))
    relevance_score = Column(Float)  # How relevant to the stock

    created_at = Column(DateTime, default=func.now())

    # Relationships
    stock = relationship("Stock")
