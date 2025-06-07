import asyncio
import aiohttp
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import json

from agents.base_agent import BaseAgent, Task
from utils.config import config
from data.database import stock_operations

class ResearchAgent(BaseAgent):
    """Agent responsible for collecting stock market data from various sources"""
    
    def __init__(self):
        super().__init__("research_agent", "Stock Data Research Agent")
        
        self.capabilities = [
            "fetch_stock_price",
            "fetch_historical_data", 
            "fetch_company_info",
            "fetch_financial_data",
            "fetch_market_news"
        ]
        
        # Rate limiting
        self.last_api_call = {}
        self.min_api_interval = 60 / config.api.api_rate_limit  # seconds between calls
        
    async def process_task(self, task: Task):
        """Process a research task"""
        task_type = task.type
        data = task.data
        
        try:
            if task_type == "fetch_stock_price":
                return await self._fetch_current_price(data.get("symbol"))
            
            elif task_type == "fetch_historical_data":
                return await self._fetch_historical_data(
                    data.get("symbol"),
                    data.get("period", "1mo"),
                    data.get("interval", "1d")
                )
            
            elif task_type == "fetch_company_info":
                return await self._fetch_company_info(data.get("symbol"))
            
            elif task_type == "fetch_financial_data":
                return await self._fetch_financial_data(data.get("symbol"))
            
            elif task_type == "fetch_market_news":
                return await self._fetch_market_news(
                    data.get("symbol"),
                    data.get("count", 10)
                )
            
            elif task_type == "full_research":
                return await self._conduct_full_research(data.get("symbol"))
            
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing task {task.id}: {str(e)}")
            raise
    
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides"""
        return self.capabilities
    
    async def _rate_limit_check(self, api_name: str):
        """Check and enforce rate limiting"""
        now = datetime.now()
        if api_name in self.last_api_call:
            time_since_last = (now - self.last_api_call[api_name]).total_seconds()
            if time_since_last < self.min_api_interval:
                sleep_time = self.min_api_interval - time_since_last
                self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        self.last_api_call[api_name] = now
    
    async def _fetch_current_price(self, symbol: str):
        """Fetch current stock price"""
        self.logger.info(f"Fetching current price for {symbol}")
        
        try:
            await self._rate_limit_check("yahoo_finance")
            
            # Use yfinance for current price
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current market data
            current_data = {
                "symbol": symbol,
                "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
                "previous_close": info.get("previousClose"),
                "open": info.get("open"),
                "day_low": info.get("dayLow"),
                "day_high": info.get("dayHigh"),
                "volume": info.get("volume"),
                "market_cap": info.get("marketCap"),
                "change": None,
                "change_percent": None,
                "timestamp": datetime.now(),
                "data_source": "yahoo_finance"
            }
            
            # Calculate change if we have both prices
            if current_data["current_price"] and current_data["previous_close"]:
                change = current_data["current_price"] - current_data["previous_close"]
                change_percent = (change / current_data["previous_close"]) * 100
                current_data["change"] = change
                current_data["change_percent"] = change_percent
            
            self.logger.info(f"Successfully fetched price for {symbol}: {current_data['current_price']}")
            return current_data
            
        except Exception as e:
            self.logger.error(f"Error fetching current price for {symbol}: {str(e)}")
            raise
    
    async def _fetch_historical_data(self, symbol: str, period: str = "1mo", interval: str = "1d"):
        """Fetch historical stock data"""
        self.logger.info(f"Fetching historical data for {symbol} (period: {period}, interval: {interval})")
        
        try:
            await self._rate_limit_check("yahoo_finance")
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                raise ValueError(f"No historical data found for {symbol}")
            
            # Convert to list of dictionaries
            historical_data = []
            for date, row in hist.iterrows():
                historical_data.append({
                    "date": date.to_pydatetime(),
                    "open_price": float(row["Open"]),
                    "high_price": float(row["High"]),
                    "low_price": float(row["Low"]),
                    "close_price": float(row["Close"]),
                    "volume": int(row["Volume"]),
                    "data_source": "yahoo_finance"
                })
            
            # Save to database
            if stock_operations:
                await asyncio.get_event_loop().run_in_executor(
                    None, stock_operations.save_price_data, symbol, historical_data
                )
            
            result = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "data_count": len(historical_data),
                "data": historical_data,
                "timestamp": datetime.now()
            }
            
            self.logger.info(f"Successfully fetched {len(historical_data)} historical records for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            raise
    
    async def _fetch_company_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch company information"""
        self.logger.info(f"Fetching company info for {symbol}")
        
        try:
            await self._rate_limit_check("yahoo_finance")
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            company_data = {
                "symbol": symbol,
                "company_name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "business_summary": info.get("businessSummary", ""),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "float_shares": info.get("floatShares"),
                "website": info.get("website", ""),
                "exchange": info.get("exchange", ""),
                "currency": info.get("currency", ""),
                "country": info.get("country", ""),
                "employees": info.get("fullTimeEmployees"),
                "timestamp": datetime.now(),
                "data_source": "yahoo_finance"
            }
            
            self.logger.info(f"Successfully fetched company info for {symbol}")
            return company_data
            
        except Exception as e:
            self.logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            raise
    
    async def _fetch_financial_data(self, symbol: str):
        """Fetch financial metrics and ratios"""
        self.logger.info(f"Fetching financial data for {symbol}")
        
        try:
            await self._rate_limit_check("yahoo_finance")
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            financial_data = {
                "symbol": symbol,
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "ps_ratio": info.get("priceToSalesTrailing12Months"),
                "peg_ratio": info.get("pegRatio"),
                "eps": info.get("trailingEps"),
                "forward_eps": info.get("forwardEps"),
                "dividend_yield": info.get("dividendYield"),
                "dividend_rate": info.get("dividendRate"),
                "book_value": info.get("bookValue"),
                "price_to_book": info.get("priceToBook"),
                "return_on_equity": info.get("returnOnEquity"),
                "return_on_assets": info.get("returnOnAssets"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "revenue": info.get("totalRevenue"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "beta": info.get("beta"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "timestamp": datetime.now(),
                "data_source": "yahoo_finance"
            }
            
            self.logger.info(f"Successfully fetched financial data for {symbol}")
            return financial_data
            
        except Exception as e:
            self.logger.error(f"Error fetching financial data for {symbol}: {str(e)}")
            raise
    
    async def _fetch_market_news(self, symbol: str, count: int = 10):
        """Fetch market news (general or stock-specific)"""
        self.logger.info(f"Fetching market news for {symbol if symbol else 'general market'}")
        
        try:
            await self._rate_limit_check("yahoo_finance")
            
            if symbol:
                ticker = yf.Ticker(symbol)
                news = ticker.news
            else:
                # For general market news, we'd need to implement a news API
                # For now, return empty result
                news = []
            
            news_data = []
            for article in news[:count]:
                news_item = {
                    "title": article.get("title", ""),
                    "summary": article.get("summary", ""),
                    "url": article.get("link", ""),
                    "publisher": article.get("publisher", ""),
                    "published_at": datetime.fromtimestamp(article.get("providerPublishTime", 0)),
                    "symbol": symbol,
                    "data_source": "yahoo_finance"
                }
                news_data.append(news_item)
            
            result = {
                "symbol": symbol,
                "news_count": len(news_data),
                "news": news_data,
                "timestamp": datetime.now()
            }
            
            self.logger.info(f"Successfully fetched {len(news_data)} news articles")
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching market news: {str(e)}")
            raise
    
    async def _conduct_full_research(self, symbol: str):
        """Conduct comprehensive research on a stock"""
        self.logger.info(f"Conducting full research for {symbol}")
        
        try:
            # Gather all available data
            tasks = [
                self._fetch_current_price(symbol),
                self._fetch_historical_data(symbol, "3mo", "1d"),
                self._fetch_company_info(symbol),
                self._fetch_financial_data(symbol),
                self._fetch_market_news(symbol, 5)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Compile results
            research_data = {
                "symbol": symbol,
                "research_timestamp": datetime.now(),
                "data_source": "yahoo_finance",
                "current_price": results[0] if not isinstance(results[0], Exception) else None,
                "historical_data": results[1] if not isinstance(results[1], Exception) else None,
                "company_info": results[2] if not isinstance(results[2], Exception) else None,
                "financial_data": results[3] if not isinstance(results[3], Exception) else None,
                "news": results[4] if not isinstance(results[4], Exception) else None,
                "errors": []
            }
            
            # Log any errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_msg = f"Error in task {i}: {str(result)}"
                    research_data["errors"].append(error_msg)
                    self.logger.warning(error_msg)
            
            self.logger.info(f"Completed full research for {symbol}")
            return research_data
            
        except Exception as e:
            self.logger.error(f"Error conducting full research for {symbol}: {str(e)}")
            raise