import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
import anthropic
from dataclasses import asdict

try:
    from .base_agent import BaseAgent, Task
    from ..utils.config import config
    from ..data.database import stock_operations
    from ..data.models import ResearchReport, ResearchInsight
except ImportError:
    from agents.base_agent import BaseAgent, Task
    from utils.config import config
    from data.database import stock_operations
    from data.models import ResearchReport, ResearchInsight

class AnalysisAgent(BaseAgent):
    """Agent responsible for analyzing stock data and generating insights"""
    
    def __init__(self):
        super().__init__("analysis_agent", "Stock Analysis Agent")
        
        self.capabilities = [
            "technical_analysis",
            "fundamental_analysis",
            "trend_analysis", 
            "ai_insights",
            "risk_assessment",
            "price_prediction",
            "comprehensive_analysis"
        ]
        
        # Initialize Claude client
        if config.api.anthropic_api_key:
            self.claude_client = anthropic.Anthropic(api_key=config.api.anthropic_api_key)
        else:
            self.claude_client = None
            self.logger.warning("Claude API key not provided - AI insights will be limited")
            
        # Technical analysis parameters
        self.ta_params = {
            "sma_periods": [20, 50, 200],  # Simple Moving Average periods
            "ema_periods": [12, 26],       # Exponential Moving Average periods  
            "rsi_period": 14,              # RSI period
            "bb_period": 20,               # Bollinger Bands period
            "bb_std": 2,                   # Bollinger Bands standard deviation
            "macd_fast": 12,               # MACD fast period
            "macd_slow": 26,               # MACD slow period
            "macd_signal": 9               # MACD signal period
        }

    async def process_task(self, task: Task) -> Any:
        """Process an analysis task"""
        task_type = task.type
        data = task.data
        
        try:
            if task_type == "technical_analysis":
                return await self._perform_technical_analysis(
                    data.get("symbol"),
                    data.get("period", "3mo")
                )
            
            elif task_type == "fundamental_analysis":
                return await self._perform_fundamental_analysis(
                    data.get("symbol")
                )
            
            elif task_type == "trend_analysis":
                return await self._perform_trend_analysis(
                    data.get("symbol"),
                    data.get("timeframe", "1mo")
                )
            
            elif task_type == "ai_insights":
                return await self._generate_ai_insights(
                    data.get("symbol"),
                    data.get("analysis_data")
                )
            
            elif task_type == "risk_assessment":
                return await self._assess_risk(
                    data.get("symbol")
                )
            
            elif task_type == "price_prediction":
                return await self._predict_price_movement(
                    data.get("symbol"),
                    data.get("horizon", "1w")
                )
                
            elif task_type == "comprehensive_analysis":
                return await self._conduct_comprehensive_analysis(
                    data.get("symbol")
                )
            
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing analysis task {task.id}: {str(e)}")
            raise

    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides"""
        return self.capabilities

    async def _get_stock_data(self, symbol: str, period: str = "3mo") -> Optional[pd.DataFrame]:
        """Get stock price data, request from Research Agent if needed"""
        try:
            # Import db_manager here to avoid circular imports
            from data.database import db_manager
            
            if db_manager is None:
                self.logger.error("Database manager not initialized")
                return None
            
            # First, try to get data from database
            with db_manager.get_session() as session:
                from data.models import Stock, PriceData
                
                stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                if not stock:
                    self.logger.warning(f"Stock {symbol} not found in database")
                    return None
                
                # Get price data from database
                query = session.query(PriceData).filter(
                    PriceData.stock_id == stock.id
                ).order_by(PriceData.date.asc())
                
                price_records = query.all()
                
                if not price_records:
                    self.logger.info(f"No price data found for {symbol}, requesting from Research Agent")
                    # Request data from Research Agent
                    await self._request_data_from_research_agent(symbol, period)
                    # Wait a bit and retry getting data
                    await asyncio.sleep(2)
                    price_records = query.all()
                
                if not price_records:
                    self.logger.error(f"Unable to get price data for {symbol}")
                    return None
                
                # Convert to DataFrame
                data = []
                for record in price_records:
                    data.append({
                        'Date': record.date,
                        'Open': record.open_price,
                        'High': record.high_price,
                        'Low': record.low_price,
                        'Close': record.close_price,
                        'Volume': record.volume
                    })
                
                df = pd.DataFrame(data)
                df.set_index('Date', inplace=True)
                df = df.sort_index()
                
                # Filter by period if needed
                if period:
                    cutoff_date = datetime.now() - self._parse_period(period)
                    df = df[df.index >= cutoff_date]
                
                self.logger.info(f"Retrieved {len(df)} price records for {symbol}")
                return df
                
        except Exception as e:
            self.logger.error(f"Error getting stock data for {symbol}: {str(e)}")
            return None

    def _parse_period(self, period: str) -> timedelta:
        """Parse period string to timedelta"""
        period = period.lower()
        if period.endswith('d'):
            return timedelta(days=int(period[:-1]))
        elif period.endswith('w'):
            return timedelta(weeks=int(period[:-1]))
        elif period.endswith('mo'):
            return timedelta(days=int(period[:-2]) * 30)
        elif period.endswith('y'):
            return timedelta(days=int(period[:-1]) * 365)
        else:
            return timedelta(days=90)  # Default 3 months

    async def _request_data_from_research_agent(self, symbol: str, period: str):
        """Request data from Research Agent"""
        if "research" in self.known_agents:  # Use "research" instead of "research_agent"
            research_agent = self.known_agents["research"]
            task = Task(
                type="fetch_historical_data",
                data={"symbol": symbol, "period": period},
                requester_id=self.agent_id
            )
            await research_agent.add_task(task)
            # Wait a bit for processing
            await asyncio.sleep(3)
        else:
            self.logger.warning("Research agent not found in known agents")

    async def _perform_technical_analysis(self, symbol: str, period: str = "3mo") -> Dict[str, Any]:
        """Perform comprehensive technical analysis"""
        self.logger.info(f"Performing technical analysis for {symbol}")
        
        df = await self._get_stock_data(symbol, period)
        if df is None or df.empty:
            raise ValueError(f"No data available for technical analysis of {symbol}")
        
        analysis = {
            "symbol": symbol,
            "analysis_type": "technical",
            "period": period,
            "data_points": len(df),
            "analysis_date": datetime.now(),
            "indicators": {}
        }
        
        try:
            # Calculate technical indicators
            analysis["indicators"].update(self._calculate_moving_averages(df))
            analysis["indicators"].update(self._calculate_rsi(df))
            analysis["indicators"].update(self._calculate_macd(df))
            analysis["indicators"].update(self._calculate_bollinger_bands(df))
            analysis["indicators"].update(self._calculate_volume_analysis(df))
            
            # Generate signals
            analysis["signals"] = self._generate_technical_signals(df, analysis["indicators"])
            
            # Calculate overall technical score
            analysis["technical_score"] = self._calculate_technical_score(analysis["indicators"], analysis["signals"])
            
            self.logger.info(f"Technical analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis for {symbol}: {str(e)}")
            raise

    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Simple and Exponential Moving Averages"""
        mas = {}
        
        # Simple Moving Averages
        for period in self.ta_params["sma_periods"]:
            if len(df) >= period:
                sma = df['Close'].rolling(window=period).mean()
                mas[f"sma_{period}"] = {
                    "current": float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None,
                    "previous": float(sma.iloc[-2]) if len(sma) > 1 and not pd.isna(sma.iloc[-2]) else None
                }
        
        # Exponential Moving Averages  
        for period in self.ta_params["ema_periods"]:
            if len(df) >= period:
                ema = df['Close'].ewm(span=period).mean()
                mas[f"ema_{period}"] = {
                    "current": float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else None,
                    "previous": float(ema.iloc[-2]) if len(ema) > 1 and not pd.isna(ema.iloc[-2]) else None
                }
        
        return {"moving_averages": mas}

    def _calculate_rsi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Relative Strength Index"""
        if len(df) < self.ta_params["rsi_period"] + 1:
            return {"rsi": None}
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.ta_params["rsi_period"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.ta_params["rsi_period"]).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
        
        # Determine RSI condition
        rsi_condition = "neutral"
        if current_rsi:
            if current_rsi > 70:
                rsi_condition = "overbought"
            elif current_rsi < 30:
                rsi_condition = "oversold"
        
        return {
            "rsi": {
                "current": current_rsi,
                "condition": rsi_condition,
                "period": self.ta_params["rsi_period"]
            }
        }

    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(df) < max(self.ta_params["macd_fast"], self.ta_params["macd_slow"]) + self.ta_params["macd_signal"]:
            return {"macd": None}
        
        ema_fast = df['Close'].ewm(span=self.ta_params["macd_fast"]).mean()
        ema_slow = df['Close'].ewm(span=self.ta_params["macd_slow"]).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.ta_params["macd_signal"]).mean()
        histogram = macd_line - signal_line
        
        return {
            "macd": {
                "macd_line": float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
                "signal_line": float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None,
                "histogram": float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None,
                "trend": "bullish" if histogram.iloc[-1] > 0 else "bearish" if histogram.iloc[-1] < 0 else "neutral"
            }
        }

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Bollinger Bands"""
        if len(df) < self.ta_params["bb_period"]:
            return {"bollinger_bands": None}
        
        sma = df['Close'].rolling(window=self.ta_params["bb_period"]).mean()
        std = df['Close'].rolling(window=self.ta_params["bb_period"]).std()
        
        upper_band = sma + (std * self.ta_params["bb_std"])
        lower_band = sma - (std * self.ta_params["bb_std"])
        
        current_price = df['Close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_sma = sma.iloc[-1]
        
        # Determine position
        bb_position = "middle"
        if current_price > current_upper:
            bb_position = "above_upper"
        elif current_price < current_lower:
            bb_position = "below_lower"
        elif current_price > current_sma:
            bb_position = "upper_half"
        else:
            bb_position = "lower_half"
        
        return {
            "bollinger_bands": {
                "upper_band": float(current_upper) if not pd.isna(current_upper) else None,
                "middle_band": float(current_sma) if not pd.isna(current_sma) else None,
                "lower_band": float(current_lower) if not pd.isna(current_lower) else None,
                "current_price": float(current_price),
                "position": bb_position,
                "band_width": float((current_upper - current_lower) / current_sma * 100) if not pd.isna(current_upper) else None
            }
        }

    def _calculate_volume_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns"""
        if len(df) < 20:
            return {"volume_analysis": None}
        
        avg_volume = df['Volume'].rolling(window=20).mean()
        current_volume = df['Volume'].iloc[-1]
        avg_volume_20 = avg_volume.iloc[-1]
        
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
        
        volume_trend = "normal"
        if volume_ratio > 2:
            volume_trend = "high"
        elif volume_ratio > 1.5:
            volume_trend = "above_average"
        elif volume_ratio < 0.5:
            volume_trend = "low"
        
        return {
            "volume_analysis": {
                "current_volume": int(current_volume),
                "avg_volume_20d": int(avg_volume_20) if not pd.isna(avg_volume_20) else None,
                "volume_ratio": float(volume_ratio),
                "volume_trend": volume_trend
            }
        }

    def _generate_technical_signals(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, str]:
        """Generate buy/sell/hold signals based on technical indicators"""
        signals = {}
        
        current_price = df['Close'].iloc[-1]
        
        # Moving Average signals
        ma_signals = []
        if "moving_averages" in indicators:
            mas = indicators["moving_averages"]
            for ma_key, ma_data in mas.items():
                if ma_data["current"]:
                    if current_price > ma_data["current"]:
                        ma_signals.append("bullish")
                    else:
                        ma_signals.append("bearish")
        
        if ma_signals:
            bullish_count = ma_signals.count("bullish")
            signals["moving_average"] = "bullish" if bullish_count > len(ma_signals) / 2 else "bearish"
        
        # RSI signals
        if "rsi" in indicators and indicators["rsi"]:
            rsi_data = indicators["rsi"]
            if rsi_data["condition"] == "oversold":
                signals["rsi"] = "buy"
            elif rsi_data["condition"] == "overbought":
                signals["rsi"] = "sell"
            else:
                signals["rsi"] = "hold"
        
        # MACD signals
        if "macd" in indicators and indicators["macd"]:
            macd_data = indicators["macd"]
            signals["macd"] = macd_data["trend"]
        
        # Bollinger Bands signals
        if "bollinger_bands" in indicators and indicators["bollinger_bands"]:
            bb_data = indicators["bollinger_bands"]
            if bb_data["position"] == "below_lower":
                signals["bollinger_bands"] = "buy"
            elif bb_data["position"] == "above_upper":
                signals["bollinger_bands"] = "sell"
            else:
                signals["bollinger_bands"] = "hold"
        
        return signals

    def _calculate_technical_score(self, indicators: Dict[str, Any], signals: Dict[str, str]) -> Dict[str, Any]:
        """Calculate overall technical score and recommendation"""
        
        # Count bullish/bearish signals
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        signal_weights = {
            "moving_average": 2,
            "rsi": 1,
            "macd": 2,
            "bollinger_bands": 1
        }
        
        for signal_type, signal in signals.items():
            weight = signal_weights.get(signal_type, 1)
            total_signals += weight
            
            if signal in ["bullish", "buy"]:
                bullish_signals += weight
            elif signal in ["bearish", "sell"]:
                bearish_signals += weight
        
        if total_signals == 0:
            return {"score": 0, "recommendation": "HOLD", "confidence": 0}
        
        # Calculate score (-100 to +100)
        net_score = (bullish_signals - bearish_signals) / total_signals * 100
        
        # Determine recommendation
        if net_score > 30:
            recommendation = "BUY"
        elif net_score < -30:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        # Calculate confidence (0 to 1)
        confidence = abs(net_score) / 100
        
        return {
            "score": round(net_score, 2),
            "recommendation": recommendation,
            "confidence": round(confidence, 2),
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals
        }

    async def _perform_fundamental_analysis(self, symbol: str) -> Dict[str, Any]:
        """Perform fundamental analysis using financial data"""
        self.logger.info(f"Performing fundamental analysis for {symbol}")
        
        try:
            # Get financial data from database or request from Research Agent
            financial_data = await self._get_financial_data(symbol)
            
            if not financial_data:
                # Request from Research Agent
                await self._request_financial_data(symbol)
                financial_data = await self._get_financial_data(symbol)
            
            analysis = {
                "symbol": symbol,
                "analysis_type": "fundamental",
                "analysis_date": datetime.now(),
                "financial_metrics": financial_data or {},
                "valuation": {},
                "financial_health": {},
                "growth_metrics": {}
            }
            
            if not financial_data:
                # Return analysis with limited data
                analysis["error"] = "No financial data available"
                analysis["fundamental_score"] = {
                    "score": 50.0,  # Neutral score
                    "recommendation": "HOLD",
                    "raw_score": 0,
                    "max_score": 0,
                    "note": "Unable to perform complete fundamental analysis due to missing data"
                }
                self.logger.warning(f"Limited fundamental analysis for {symbol} - no financial data")
                return analysis
            
            # Valuation analysis
            analysis["valuation"] = self._analyze_valuation(financial_data)
            
            # Financial health analysis
            analysis["financial_health"] = self._analyze_financial_health(financial_data)
            
            # Growth analysis (would need historical data)
            analysis["growth_metrics"] = self._analyze_growth(financial_data)
            
            # Overall fundamental score
            analysis["fundamental_score"] = self._calculate_fundamental_score(analysis)
            
            self.logger.info(f"Fundamental analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in fundamental analysis for {symbol}: {str(e)}")
            raise

    async def _get_financial_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get financial data from database"""
        try:
            from data.database import db_manager
            
            if db_manager is None:
                self.logger.error("Database manager not initialized")
                return None
                
            with db_manager.get_session() as session:
                from data.models import Stock, FinancialData
                
                stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                if not stock:
                    return None
                
                # Get latest financial data
                financial_record = session.query(FinancialData).filter(
                    FinancialData.stock_id == stock.id
                ).order_by(FinancialData.created_at.desc()).first()
                
                if not financial_record:
                    return None
                
                # Convert to dictionary
                financial_data = {
                    "pe_ratio": financial_record.pe_ratio,
                    "pb_ratio": financial_record.pb_ratio,
                    "debt_to_equity": financial_record.debt_to_equity,
                    "roe": financial_record.roe,
                    "roa": financial_record.roa,
                    "current_ratio": financial_record.current_ratio,
                    "quick_ratio": financial_record.quick_ratio,
                    "operating_margin": financial_record.operating_margin,
                    "profit_margin": financial_record.profit_margin,
                    "eps": financial_record.eps,
                    "revenue": financial_record.revenue,
                    "net_income": financial_record.net_income,
                    "dividend_yield": financial_record.dividend_yield
                }
                
                return financial_data
                
        except Exception as e:
            self.logger.error(f"Error getting financial data for {symbol}: {str(e)}")
            return None

    async def _request_financial_data(self, symbol: str):
        """Request financial data from Research Agent"""
        if "research" in self.known_agents:  # Use "research" instead of "research_agent"
            research_agent = self.known_agents["research"] 
            task = Task(
                type="fetch_financial_data",
                data={"symbol": symbol},
                requester_id=self.agent_id
            )
            await research_agent.add_task(task)
            await asyncio.sleep(2)
        else:
            self.logger.warning("Research agent not found in known agents")

    def _analyze_valuation(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze valuation metrics"""
        valuation = {}
        
        # P/E ratio analysis
        pe_ratio = financial_data.get("pe_ratio")
        if pe_ratio:
            if pe_ratio < 15:
                valuation["pe_assessment"] = "undervalued"
            elif pe_ratio > 25:
                valuation["pe_assessment"] = "overvalued"
            else:
                valuation["pe_assessment"] = "fairly_valued"
            valuation["pe_ratio"] = pe_ratio
        
        # P/B ratio analysis
        pb_ratio = financial_data.get("pb_ratio") 
        if pb_ratio:
            if pb_ratio < 1:
                valuation["pb_assessment"] = "undervalued"
            elif pb_ratio > 3:
                valuation["pb_assessment"] = "overvalued"
            else:
                valuation["pb_assessment"] = "fairly_valued"
            valuation["pb_ratio"] = pb_ratio
        
        return valuation

    def _analyze_financial_health(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial health metrics"""
        health = {}
        
        # Debt analysis
        debt_to_equity = financial_data.get("debt_to_equity")
        if debt_to_equity is not None:
            if debt_to_equity < 0.3:
                health["debt_level"] = "low"
            elif debt_to_equity > 0.7:
                health["debt_level"] = "high"
            else:
                health["debt_level"] = "moderate"
        
        # Liquidity analysis
        current_ratio = financial_data.get("current_ratio")
        if current_ratio:
            if current_ratio > 2:
                health["liquidity"] = "strong"
            elif current_ratio > 1:
                health["liquidity"] = "adequate"
            else:
                health["liquidity"] = "weak"
        
        # Profitability analysis  
        roe = financial_data.get("roe")
        if roe:
            if roe > 0.15:
                health["profitability"] = "strong"
            elif roe > 0.1:
                health["profitability"] = "adequate"
            else:
                health["profitability"] = "weak"
        
        return health

    def _analyze_growth(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze growth metrics (simplified without historical data)"""
        # This would be enhanced with historical data comparison
        growth = {
            "eps": financial_data.get("eps"),
            "revenue": financial_data.get("revenue"),
            "note": "Growth analysis requires historical data comparison"
        }
        return growth

    def _calculate_fundamental_score(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall fundamental score"""
        score = 0
        max_score = 0
        
        # Valuation score
        valuation = analysis.get("valuation", {})
        if valuation.get("pe_assessment") == "undervalued":
            score += 2
        elif valuation.get("pe_assessment") == "fairly_valued":
            score += 1
        max_score += 2
        
        if valuation.get("pb_assessment") == "undervalued":
            score += 2
        elif valuation.get("pb_assessment") == "fairly_valued":
            score += 1
        max_score += 2
        
        # Financial health score
        health = analysis.get("financial_health", {})
        if health.get("debt_level") == "low":
            score += 2
        elif health.get("debt_level") == "moderate":
            score += 1
        max_score += 2
        
        if health.get("liquidity") == "strong":
            score += 2
        elif health.get("liquidity") == "adequate":
            score += 1
        max_score += 2
        
        if health.get("profitability") == "strong":
            score += 2
        elif health.get("profitability") == "adequate":
            score += 1
        max_score += 2
        
        # Calculate percentage score
        percentage_score = (score / max_score * 100) if max_score > 0 else 0
        
        # Determine recommendation
        if percentage_score > 70:
            recommendation = "BUY"
        elif percentage_score > 40:
            recommendation = "HOLD"
        else:
            recommendation = "SELL"
        
        return {
            "score": round(percentage_score, 2),
            "recommendation": recommendation,
            "raw_score": score,
            "max_score": max_score
        }

    async def _conduct_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """Conduct comprehensive analysis combining technical and fundamental"""
        self.logger.info(f"Conducting comprehensive analysis for {symbol}")
        
        try:
            # Run both technical and fundamental analysis
            tasks = [
                self._perform_technical_analysis(symbol),
                self._perform_fundamental_analysis(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            technical_analysis = results[0] if not isinstance(results[0], Exception) else None
            fundamental_analysis = results[1] if not isinstance(results[1], Exception) else None
            
            # Combine analyses
            comprehensive = {
                "symbol": symbol,
                "analysis_type": "comprehensive",
                "analysis_date": datetime.now(),
                "technical_analysis": technical_analysis,
                "fundamental_analysis": fundamental_analysis,
                "combined_recommendation": {},
                "risk_factors": [],
                "opportunities": []
            }
            
            # Generate combined recommendation
            comprehensive["combined_recommendation"] = self._combine_recommendations(
                technical_analysis, fundamental_analysis
            )
            
            # Generate AI insights if Claude is available
            if self.claude_client:
                ai_insights = await self._generate_ai_insights(symbol, comprehensive)
                comprehensive["ai_insights"] = ai_insights
            
            # Save to database
            await self._save_analysis_to_database(symbol, comprehensive)
            
            self.logger.info(f"Comprehensive analysis completed for {symbol}")
            return comprehensive
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis for {symbol}: {str(e)}")
            raise

    def _combine_recommendations(self, technical: Optional[Dict], fundamental: Optional[Dict]) -> Dict[str, Any]:
        """Combine technical and fundamental recommendations"""
        
        if not technical and not fundamental:
            return {"recommendation": "HOLD", "confidence": 0, "reasoning": "Insufficient data"}
        
        tech_score = 0
        fund_score = 0
        
        if technical and "technical_score" in technical:
            tech_score = technical["technical_score"]["score"]
        
        if fundamental and "fundamental_score" in fundamental:
            fund_score = fundamental["fundamental_score"]["score"]
        
        # Weight the scores (60% fundamental, 40% technical for long-term)
        combined_score = (fund_score * 0.6) + (tech_score * 0.4)
        
        # Determine final recommendation
        if combined_score > 60:
            recommendation = "BUY"
        elif combined_score > 30:
            recommendation = "HOLD"
        else:
            recommendation = "SELL"
        
        # Calculate confidence
        confidence = min(abs(combined_score) / 100, 1.0)
        
        return {
            "recommendation": recommendation,
            "confidence": round(confidence, 2),
            "combined_score": round(combined_score, 2),
            "technical_score": tech_score,
            "fundamental_score": fund_score,
            "reasoning": f"Combined analysis based on technical ({tech_score:.1f}) and fundamental ({fund_score:.1f}) scores"
        }

    async def _generate_ai_insights(self, symbol: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered insights using Claude"""
        if not self.claude_client:
            return {"error": "Claude API not available"}
        
        try:
            # Prepare prompt for Claude
            prompt = self._prepare_analysis_prompt(symbol, analysis_data)
            
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            ai_content = response.content[0].text
            
            return {
                "ai_summary": ai_content,
                "generated_at": datetime.now(),
                "model": "claude-3-5-sonnet-20241022"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating AI insights: {str(e)}")
            return {"error": f"AI insights generation failed: {str(e)}"}

    def _prepare_analysis_prompt(self, symbol: str, analysis_data: Dict[str, Any]) -> str:
        """Prepare prompt for Claude analysis"""
        
        prompt = f"""
        Analyze the following stock data for {symbol} and provide insights:

        Technical Analysis:
        {json.dumps(analysis_data.get('technical_analysis', {}), indent=2, default=str)}

        Fundamental Analysis:
        {json.dumps(analysis_data.get('fundamental_analysis', {}), indent=2, default=str)}

        Please provide:
        1. Key strengths and weaknesses
        2. Risk factors to consider
        3. Potential opportunities
        4. Short-term and long-term outlook
        5. Specific recommendations for investors

        Keep the response concise but comprehensive, focusing on actionable insights.
        """
        
        return prompt

    async def _save_analysis_to_database(self, symbol: str, analysis: Dict[str, Any]):
        """Save analysis results to database"""
        try:
            from data.database import db_manager
            
            if db_manager is None:
                self.logger.error("Database manager not initialized - cannot save analysis")
                return
                
            with db_manager.get_session() as session:
                from data.models import Stock, ResearchReport, ResearchInsight
                
                stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                if not stock:
                    self.logger.warning(f"Stock {symbol} not found for saving analysis")
                    return
                
                # Create research report
                report = ResearchReport(
                    stock_id=stock.id,
                    report_type=analysis.get("analysis_type", "comprehensive"),
                    title=f"Analysis Report for {symbol}",
                    summary=analysis.get("combined_recommendation", {}).get("reasoning", ""),
                    detailed_analysis=json.dumps(analysis, default=str),
                    recommendation=analysis.get("combined_recommendation", {}).get("recommendation", "HOLD"),
                    confidence_score=analysis.get("combined_recommendation", {}).get("confidence", 0),
                    generated_by_agent=self.agent_id,
                    analysis_date=analysis.get("analysis_date", datetime.now()),
                    data_sources=json.dumps(["technical_analysis", "fundamental_analysis"])
                )
                
                session.add(report)
                session.flush()  # Get the report ID
                
                # Create insights
                self._create_analysis_insights(session, report.id, analysis)
                
                self.logger.info(f"Analysis saved to database for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error saving analysis to database: {str(e)}")

    def _create_analysis_insights(self, session, report_id: int, analysis: Dict[str, Any]):
        """Create individual insights from analysis"""
        insights = []
        
        # Technical insights
        if "technical_analysis" in analysis and analysis["technical_analysis"]:
            tech = analysis["technical_analysis"]
            if "technical_score" in tech:
                insight = ResearchInsight(
                    report_id=report_id,
                    insight_type="technical",
                    title="Technical Analysis Score",
                    description=f"Technical score: {tech['technical_score']['score']}, Recommendation: {tech['technical_score']['recommendation']}",
                    impact_score=tech['technical_score']['score'] / 100,
                    confidence_level=tech['technical_score']['confidence']
                )
                insights.append(insight)
        
        # Fundamental insights
        if "fundamental_analysis" in analysis and analysis["fundamental_analysis"]:
            fund = analysis["fundamental_analysis"]
            if "fundamental_score" in fund:
                insight = ResearchInsight(
                    report_id=report_id,
                    insight_type="fundamental",
                    title="Fundamental Analysis Score",
                    description=f"Fundamental score: {fund['fundamental_score']['score']}, Recommendation: {fund['fundamental_score']['recommendation']}",
                    impact_score=fund['fundamental_score']['score'] / 100,
                    confidence_level=fund['fundamental_score']['score'] / 100
                )
                insights.append(insight)
        
        # Add insights to session
        for insight in insights:
            session.add(insight)

    async def _perform_trend_analysis(self, symbol: str, timeframe: str = "1mo") -> Dict[str, Any]:
        """Analyze price trends over specified timeframe"""
        self.logger.info(f"Performing trend analysis for {symbol}")
        
        df = await self._get_stock_data(symbol, timeframe)
        if df is None or df.empty:
            raise ValueError(f"No data available for trend analysis of {symbol}")
        
        # Calculate trend metrics
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        high_price = df['High'].max()
        low_price = df['Low'].min()
        
        # Price change metrics
        total_return = ((end_price - start_price) / start_price) * 100
        volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized
        
        # Trend direction
        if total_return > 5:
            trend_direction = "strong_uptrend"
        elif total_return > 0:
            trend_direction = "uptrend"
        elif total_return > -5:
            trend_direction = "downtrend"
        else:
            trend_direction = "strong_downtrend"
        
        # Support and resistance levels
        recent_high = df['High'].tail(10).max()
        recent_low = df['Low'].tail(10).min()
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_date": datetime.now(),
            "price_metrics": {
                "start_price": float(start_price),
                "end_price": float(end_price),
                "high_price": float(high_price),
                "low_price": float(low_price),
                "total_return": round(total_return, 2),
                "volatility": round(volatility, 2)
            },
            "trend_analysis": {
                "direction": trend_direction,
                "strength": abs(total_return),
                "consistency": self._calculate_trend_consistency(df)
            },
            "support_resistance": {
                "recent_resistance": float(recent_high),
                "recent_support": float(recent_low),
                "current_position": "near_resistance" if end_price > (recent_high * 0.95) else "near_support" if end_price < (recent_low * 1.05) else "middle"
            }
        }

    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """Calculate how consistent the trend is (0 to 1)"""
        if len(df) < 5:
            return 0.5
        
        # Calculate rolling 5-day returns
        returns = df['Close'].pct_change(5).dropna()
        
        if len(returns) == 0:
            return 0.5
        
        # Count how many periods trend in same direction
        positive_returns = (returns > 0).sum()
        negative_returns = (returns < 0).sum()
        
        # Consistency is the proportion of periods trending in dominant direction
        total_periods = len(returns)
        dominant_direction = max(positive_returns, negative_returns)
        
        return dominant_direction / total_periods

    async def _assess_risk(self, symbol: str) -> Dict[str, Any]:
        """Assess various risk factors for the stock"""
        self.logger.info(f"Assessing risk for {symbol}")
        
        risk_assessment = {
            "symbol": symbol,
            "assessment_date": datetime.now(),
            "risk_factors": {},
            "overall_risk": "medium"
        }
        
        try:
            # Get price data for volatility analysis
            df = await self._get_stock_data(symbol, "6mo")
            if df is not None and not df.empty:
                # Price volatility risk
                returns = df['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
                
                if volatility > 40:
                    volatility_risk = "high"
                elif volatility > 20:
                    volatility_risk = "medium"
                else:
                    volatility_risk = "low"
                
                risk_assessment["risk_factors"]["volatility"] = {
                    "level": volatility_risk,
                    "value": round(volatility, 2),
                    "description": f"Annualized volatility of {volatility:.1f}%"
                }
                
                # Drawdown risk
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min() * 100
                
                if max_drawdown < -30:
                    drawdown_risk = "high"
                elif max_drawdown < -15:
                    drawdown_risk = "medium"
                else:
                    drawdown_risk = "low"
                
                risk_assessment["risk_factors"]["drawdown"] = {
                    "level": drawdown_risk,
                    "max_drawdown": round(max_drawdown, 2),
                    "description": f"Maximum drawdown of {max_drawdown:.1f}%"
                }
            
            # Get fundamental data for financial risk
            financial_data = await self._get_financial_data(symbol)
            if financial_data:
                # Debt risk
                debt_to_equity = financial_data.get("debt_to_equity")
                if debt_to_equity is not None:
                    if debt_to_equity > 1.0:
                        debt_risk = "high"
                    elif debt_to_equity > 0.5:
                        debt_risk = "medium"
                    else:
                        debt_risk = "low"
                    
                    risk_assessment["risk_factors"]["debt"] = {
                        "level": debt_risk,
                        "debt_to_equity": debt_to_equity,
                        "description": f"Debt-to-equity ratio of {debt_to_equity:.2f}"
                    }
                
                # Liquidity risk
                current_ratio = financial_data.get("current_ratio")
                if current_ratio is not None:
                    if current_ratio < 1.0:
                        liquidity_risk = "high"
                    elif current_ratio < 1.5:
                        liquidity_risk = "medium"
                    else:
                        liquidity_risk = "low"
                    
                    risk_assessment["risk_factors"]["liquidity"] = {
                        "level": liquidity_risk,
                        "current_ratio": current_ratio,
                        "description": f"Current ratio of {current_ratio:.2f}"
                    }
            
            # Calculate overall risk
            risk_levels = [factor["level"] for factor in risk_assessment["risk_factors"].values()]
            high_risks = risk_levels.count("high")
            medium_risks = risk_levels.count("medium")
            
            if high_risks > 1:
                risk_assessment["overall_risk"] = "high"
            elif high_risks > 0 or medium_risks > 2:
                risk_assessment["overall_risk"] = "medium"
            else:
                risk_assessment["overall_risk"] = "low"
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing risk for {symbol}: {str(e)}")
            risk_assessment["error"] = str(e)
            return risk_assessment

    async def _predict_price_movement(self, symbol: str, horizon: str = "1w") -> Dict[str, Any]:
        """Predict price movement based on technical indicators"""
        self.logger.info(f"Predicting price movement for {symbol} over {horizon}")
        
        try:
            # Get recent technical analysis
            technical_analysis = await self._perform_technical_analysis(symbol, "3mo")
            
            prediction = {
                "symbol": symbol,
                "horizon": horizon,
                "prediction_date": datetime.now(),
                "confidence": 0.5,
                "direction": "neutral",
                "factors": []
            }
            
            if not technical_analysis:
                prediction["error"] = "Insufficient data for prediction"
                return prediction
            
            # Analyze technical signals for prediction
            signals = technical_analysis.get("signals", {})
            indicators = technical_analysis.get("indicators", {})
            
            bullish_factors = 0
            bearish_factors = 0
            total_factors = 0
            
            # Moving average signals
            if signals.get("moving_average") == "bullish":
                bullish_factors += 2
                prediction["factors"].append("Moving averages trending upward")
            elif signals.get("moving_average") == "bearish":
                bearish_factors += 2
                prediction["factors"].append("Moving averages trending downward")
            total_factors += 2
            
            # RSI signals
            if signals.get("rsi") == "buy":
                bullish_factors += 1
                prediction["factors"].append("RSI indicates oversold condition")
            elif signals.get("rsi") == "sell":
                bearish_factors += 1
                prediction["factors"].append("RSI indicates overbought condition")
            total_factors += 1
            
            # MACD signals
            if signals.get("macd") == "bullish":
                bullish_factors += 1
                prediction["factors"].append("MACD showing bullish momentum")
            elif signals.get("macd") == "bearish":
                bearish_factors += 1
                prediction["factors"].append("MACD showing bearish momentum")
            total_factors += 1
            
            # Volume analysis
            volume_data = indicators.get("volume_analysis")
            if volume_data and volume_data.get("volume_trend") in ["high", "above_average"]:
                prediction["factors"].append(f"Volume trend: {volume_data['volume_trend']}")
                # High volume can amplify the existing trend
                if bullish_factors > bearish_factors:
                    bullish_factors += 0.5
                else:
                    bearish_factors += 0.5
            
            # Calculate prediction
            if total_factors > 0:
                net_score = (bullish_factors - bearish_factors) / total_factors
                
                if net_score > 0.3:
                    prediction["direction"] = "bullish"
                    prediction["confidence"] = min(net_score, 0.8)
                elif net_score < -0.3:
                    prediction["direction"] = "bearish"
                    prediction["confidence"] = min(abs(net_score), 0.8)
                else:
                    prediction["direction"] = "neutral"
                    prediction["confidence"] = 0.3
            
            # Add disclaimer
            prediction["disclaimer"] = "This prediction is based on technical analysis and should not be considered financial advice"
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting price movement for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "horizon": horizon,
                "error": str(e),
                "prediction_date": datetime.now()
            }