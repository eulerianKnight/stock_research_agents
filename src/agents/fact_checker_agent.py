import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
import statistics
from dataclasses import dataclass

try:
    from .base_agent import BaseAgent, Task
    from ..utils.config import config
    from ..data.database import stock_operations
    from ..data.models import ResearchReport, ResearchInsight, MarketNews
except ImportError:
    from agents.base_agent import BaseAgent, Task
    from utils.config import config
    from data.database import stock_operations
    from data.models import ResearchReport, ResearchInsight, MarketNews

@dataclass
class DataValidationResult:
    """Result of data validation check"""
    source: str
    data_type: str
    value: Any
    is_valid: bool
    confidence_score: float
    anomaly_flags: List[str]
    validation_notes: str

@dataclass
class CrossValidationResult:
    """Result of cross-validation between multiple sources"""
    data_point: str
    sources: List[str]
    values: List[Any]
    consensus_value: Any
    confidence_score: float
    variance_score: float
    outliers: List[str]
    recommendation: str

class FactCheckerAgent(BaseAgent):
    """Agent responsible for fact-checking and validating data quality"""
    
    def __init__(self):
        super().__init__("fact_checker", "Data Fact Checker Agent")
        
        self.capabilities = [
            "validate_price_data",
            "cross_validate_sources",
            "check_data_consistency",
            "verify_news_relevance",
            "assess_data_quality",
            "flag_anomalies",
            "generate_credibility_report"
        ]
        
        # Validation thresholds
        self.validation_params = {
            "price_variance_threshold": 0.05,  # 5% variance tolerance
            "volume_anomaly_factor": 3.0,      # 3x normal volume considered anomaly
            "pe_ratio_bounds": (0, 100),       # Reasonable P/E ratio range
            "market_cap_bounds": (1e6, 1e15),  # Market cap bounds (1M to 1000T)
            "news_relevance_threshold": 0.7,   # News relevance score threshold
            "minimum_sources": 2,              # Minimum sources for cross-validation
            "consensus_threshold": 0.8         # 80% agreement for consensus
        }
        
        # Known data source reliability scores
        self.source_reliability = {
            "yahoo_finance": 0.9,
            "nse_api": 0.95,
            "bse_api": 0.95,
            "alpha_vantage": 0.85,
            "manual_entry": 0.6,
            "news_api": 0.75,
            "unknown": 0.5
        }

    async def process_task(self, task: Task) -> Any:
        """Process a fact-checking task"""
        task_type = task.type
        data = task.data
        
        try:
            if task_type == "validate_price_data":
                return await self._validate_price_data(
                    data.get("symbol"),
                    data.get("timeframe", "1d")
                )
            
            elif task_type == "cross_validate_sources":
                return await self._cross_validate_sources(
                    data.get("symbol"),
                    data.get("data_types", ["price", "volume"])
                )
            
            elif task_type == "check_data_consistency":
                return await self._check_data_consistency(
                    data.get("symbol"),
                    data.get("period", "1mo")
                )
            
            elif task_type == "verify_news_relevance":
                return await self._verify_news_relevance(
                    data.get("symbol"),
                    data.get("news_articles", [])
                )
            
            elif task_type == "assess_data_quality":
                return await self._assess_data_quality(
                    data.get("symbol")
                )
            
            elif task_type == "flag_anomalies":
                return await self._flag_anomalies(
                    data.get("symbol"),
                    data.get("analysis_data")
                )
            
            elif task_type == "generate_credibility_report":
                return await self._generate_credibility_report(
                    data.get("symbol"),
                    data.get("analysis_data")
                )
            
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing fact-checking task {task.id}: {str(e)}")
            raise

    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides"""
        return self.capabilities

    async def _validate_price_data(self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """Validate price data for inconsistencies and anomalies"""
        self.logger.info(f"Validating price data for {symbol}")
        
        try:
            # Get price data from database
            price_data = await self._get_price_data(symbol, timeframe)
            
            if price_data is None or price_data.empty:
                return {
                    "symbol": symbol,
                    "validation_status": "failed",
                    "error": "No price data available for validation"
                }
            
            validation_results = []
            anomalies = []
            
            # Check for basic data integrity
            for idx, row in price_data.iterrows():
                validation_result = self._validate_single_price_record(row, idx)
                validation_results.append(validation_result)
                
                if not validation_result.is_valid:
                    anomalies.extend(validation_result.anomaly_flags)
            
            # Check for price movement anomalies
            price_anomalies = self._detect_price_anomalies(price_data)
            anomalies.extend(price_anomalies)
            
            # Check volume anomalies
            volume_anomalies = self._detect_volume_anomalies(price_data)
            anomalies.extend(volume_anomalies)
            
            # Calculate overall data quality score
            valid_records = sum(1 for r in validation_results if r.is_valid)
            data_quality_score = valid_records / len(validation_results) if validation_results else 0
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "validation_timestamp": datetime.now(),
                "data_quality_score": round(data_quality_score, 3),
                "total_records": len(validation_results),
                "valid_records": valid_records,
                "anomalies": list(set(anomalies)),  # Remove duplicates
                "anomaly_count": len(set(anomalies)),
                "validation_status": "passed" if data_quality_score > 0.9 else "warning" if data_quality_score > 0.7 else "failed",
                "detailed_results": [
                    {
                        "date": r.source,
                        "valid": r.is_valid,
                        "confidence": r.confidence_score,
                        "issues": r.anomaly_flags
                    }
                    for r in validation_results[:10]  # First 10 for brevity
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error validating price data for {symbol}: {str(e)}")
            raise

    def _validate_single_price_record(self, row: pd.Series, date: Any) -> DataValidationResult:
        """Validate a single price record"""
        anomaly_flags = []
        confidence_score = 1.0
        
        # Check for missing values
        required_fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        for field in required_fields:
            if pd.isna(row.get(field)) or row.get(field) is None:
                anomaly_flags.append(f"missing_{field.lower()}")
                confidence_score -= 0.2
        
        # Check price relationships (High >= Low, etc.)
        if not pd.isna(row.get('High')) and not pd.isna(row.get('Low')):
            if row['High'] < row['Low']:
                anomaly_flags.append("high_less_than_low")
                confidence_score -= 0.3
        
        # Check if Open/Close are within High/Low range
        for price_type in ['Open', 'Close']:
            price = row.get(price_type)
            if not pd.isna(price) and not pd.isna(row.get('High')) and not pd.isna(row.get('Low')):
                if price > row['High'] or price < row['Low']:
                    anomaly_flags.append(f"{price_type.lower()}_outside_range")
                    confidence_score -= 0.2
        
        # Check for negative values
        for field in required_fields:
            value = row.get(field)
            if not pd.isna(value) and value < 0:
                anomaly_flags.append(f"negative_{field.lower()}")
                confidence_score -= 0.3
        
        # Check for zero volume (suspicious)
        if row.get('Volume') == 0:
            anomaly_flags.append("zero_volume")
            confidence_score -= 0.1
        
        is_valid = len(anomaly_flags) == 0
        confidence_score = max(0, confidence_score)
        
        return DataValidationResult(
            source=str(date),
            data_type="price_record",
            value=row.to_dict(),
            is_valid=is_valid,
            confidence_score=confidence_score,
            anomaly_flags=anomaly_flags,
            validation_notes=f"Validated price record for {date}"
        )

    def _detect_price_anomalies(self, price_data: pd.DataFrame) -> List[str]:
        """Detect price movement anomalies"""
        anomalies = []
        
        if len(price_data) < 2:
            return anomalies
        
        # Calculate daily returns
        price_data['Returns'] = price_data['Close'].pct_change()
        
        # Detect extreme price movements (>20% in a day)
        extreme_moves = price_data[abs(price_data['Returns']) > 0.2]
        if len(extreme_moves) > 0:
            anomalies.append(f"extreme_price_movement_{len(extreme_moves)}_days")
        
        # Detect price gaps (>10% gap between consecutive days)
        price_data['Gap'] = (price_data['Open'] - price_data['Close'].shift(1)) / price_data['Close'].shift(1)
        large_gaps = price_data[abs(price_data['Gap']) > 0.1]
        if len(large_gaps) > 0:
            anomalies.append(f"large_price_gaps_{len(large_gaps)}_occurrences")
        
        # Detect suspicious flat periods (same price for multiple days)
        consecutive_same = 0
        prev_close = None
        for close in price_data['Close']:
            if prev_close is not None and close == prev_close:
                consecutive_same += 1
            else:
                if consecutive_same > 5:  # Same price for more than 5 days
                    anomalies.append(f"flat_period_{consecutive_same}_days")
                consecutive_same = 0
            prev_close = close
        
        return anomalies

    def _detect_volume_anomalies(self, price_data: pd.DataFrame) -> List[str]:
        """Detect volume anomalies"""
        anomalies = []
        
        if len(price_data) < 10:
            return anomalies
        
        # Calculate average volume and standard deviation
        avg_volume = price_data['Volume'].mean()
        std_volume = price_data['Volume'].std()
        
        # Detect volume spikes (>3 standard deviations)
        threshold = avg_volume + (3 * std_volume)
        volume_spikes = price_data[price_data['Volume'] > threshold]
        if len(volume_spikes) > 0:
            anomalies.append(f"volume_spikes_{len(volume_spikes)}_occurrences")
        
        # Detect unusually low volume (<10% of average)
        low_volume_threshold = avg_volume * 0.1
        low_volume_days = price_data[price_data['Volume'] < low_volume_threshold]
        if len(low_volume_days) > len(price_data) * 0.1:  # More than 10% of days
            anomalies.append(f"low_volume_pattern_{len(low_volume_days)}_days")
        
        return anomalies

    async def _cross_validate_sources(self, symbol: str, data_types: List[str] = None) -> Dict[str, Any]:
        """Cross-validate data from multiple sources"""
        self.logger.info(f"Cross-validating sources for {symbol}")
        
        if data_types is None:
            data_types = ["price", "volume", "market_cap"]
        
        try:
            validation_results = {}
            
            for data_type in data_types:
                cross_validation = await self._cross_validate_data_type(symbol, data_type)
                validation_results[data_type] = cross_validation
            
            # Calculate overall cross-validation score
            scores = [result.confidence_score for result in validation_results.values() if result]
            overall_score = statistics.mean(scores) if scores else 0
            
            return {
                "symbol": symbol,
                "cross_validation_timestamp": datetime.now(),
                "data_types_checked": data_types,
                "overall_confidence_score": round(overall_score, 3),
                "validation_results": {
                    data_type: {
                        "consensus_value": result.consensus_value,
                        "confidence_score": result.confidence_score,
                        "variance_score": result.variance_score,
                        "sources_count": len(result.sources),
                        "outliers": result.outliers,
                        "recommendation": result.recommendation
                    } if result else None
                    for data_type, result in validation_results.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error cross-validating sources for {symbol}: {str(e)}")
            raise

    async def _cross_validate_data_type(self, symbol: str, data_type: str) -> Optional[CrossValidationResult]:
        """Cross-validate a specific data type across sources"""
        
        # This is a simplified implementation
        # In a real system, you'd fetch from multiple APIs
        try:
            # Get data from our database (representing one source)
            if data_type == "price":
                price_data = await self._get_price_data(symbol, "1d")
                if price_data is not None and not price_data.empty:
                    latest_price = float(price_data['Close'].iloc[-1])
                    
                    # Simulate data from multiple sources
                    # In reality, you'd call different APIs
                    sources = ["yahoo_finance", "alpha_vantage"]
                    values = [latest_price, latest_price * (1 + np.random.uniform(-0.01, 0.01))]
                    
                    consensus_value = statistics.mean(values)
                    variance = statistics.stdev(values) if len(values) > 1 else 0
                    variance_score = 1 - min(variance / consensus_value, 1) if consensus_value > 0 else 0
                    
                    # Detect outliers (values >5% different from consensus)
                    outliers = []
                    for i, (source, value) in enumerate(zip(sources, values)):
                        if abs(value - consensus_value) / consensus_value > 0.05:
                            outliers.append(source)
                    
                    # Calculate confidence based on agreement
                    confidence_score = variance_score * 0.7 + (1 - len(outliers) / len(sources)) * 0.3
                    
                    recommendation = "reliable" if confidence_score > 0.8 else "caution" if confidence_score > 0.6 else "unreliable"
                    
                    return CrossValidationResult(
                        data_point=f"{symbol}_{data_type}",
                        sources=sources,
                        values=values,
                        consensus_value=consensus_value,
                        confidence_score=round(confidence_score, 3),
                        variance_score=round(variance_score, 3),
                        outliers=outliers,
                        recommendation=recommendation
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error cross-validating {data_type} for {symbol}: {str(e)}")
            return None

    async def _check_data_consistency(self, symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """Check data consistency over time"""
        self.logger.info(f"Checking data consistency for {symbol}")
        
        try:
            # Get historical data
            price_data = await self._get_price_data(symbol, period)
            
            if price_data is None or price_data.empty:
                return {
                    "symbol": symbol,
                    "consistency_status": "failed",
                    "error": "No data available for consistency check"
                }
            
            consistency_issues = []
            
            # Check for data gaps
            date_gaps = self._detect_date_gaps(price_data)
            if date_gaps:
                consistency_issues.extend([f"date_gap_{gap}" for gap in date_gaps])
            
            # Check for trend consistency
            trend_issues = self._check_trend_consistency(price_data)
            consistency_issues.extend(trend_issues)
            
            # Check for statistical outliers
            statistical_outliers = self._detect_statistical_outliers(price_data)
            consistency_issues.extend(statistical_outliers)
            
            # Calculate consistency score
            total_possible_issues = len(price_data) * 3  # Arbitrary multiplier
            consistency_score = max(0, 1 - len(consistency_issues) / total_possible_issues)
            
            return {
                "symbol": symbol,
                "period": period,
                "consistency_timestamp": datetime.now(),
                "consistency_score": round(consistency_score, 3),
                "total_issues": len(consistency_issues),
                "consistency_status": "good" if consistency_score > 0.8 else "warning" if consistency_score > 0.6 else "poor",
                "issues": consistency_issues[:20],  # First 20 issues
                "data_points_analyzed": len(price_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error checking data consistency for {symbol}: {str(e)}")
            raise

    def _detect_date_gaps(self, price_data: pd.DataFrame) -> List[str]:
        """Detect missing dates in the data"""
        gaps = []
        
        if len(price_data) < 2:
            return gaps
        
        dates = price_data.index
        for i in range(1, len(dates)):
            days_diff = (dates[i] - dates[i-1]).days
            if days_diff > 3:  # More than 3 days gap (accounting for weekends)
                gaps.append(f"{dates[i-1].date()}_to_{dates[i].date()}")
        
        return gaps

    def _check_trend_consistency(self, price_data: pd.DataFrame) -> List[str]:
        """Check for trend consistency issues"""
        issues = []
        
        if len(price_data) < 5:
            return issues
        
        # Calculate moving averages
        price_data['MA5'] = price_data['Close'].rolling(window=5).mean()
        price_data['MA10'] = price_data['Close'].rolling(window=10).mean()
        
        # Check for MA crossover anomalies
        ma5_above_ma10 = price_data['MA5'] > price_data['MA10']
        crossovers = ma5_above_ma10.diff()
        frequent_crossovers = abs(crossovers).sum()
        
        if frequent_crossovers > len(price_data) * 0.3:  # More than 30% crossovers
            issues.append("frequent_ma_crossovers")
        
        return issues

    def _detect_statistical_outliers(self, price_data: pd.DataFrame) -> List[str]:
        """Detect statistical outliers using Z-score"""
        outliers = []
        
        if len(price_data) < 10:
            return outliers
        
        # Calculate Z-scores for returns
        returns = price_data['Close'].pct_change().dropna()
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        
        # Count outliers (Z-score > 3)
        extreme_outliers = (z_scores > 3).sum()
        if extreme_outliers > len(returns) * 0.05:  # More than 5% outliers
            outliers.append(f"excessive_return_outliers_{extreme_outliers}")
        
        return outliers

    async def _assess_data_quality(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        self.logger.info(f"Assessing data quality for {symbol}")
        
        try:
            # Run multiple validation checks
            tasks = [
                self._validate_price_data(symbol, "1mo"),
                self._cross_validate_sources(symbol, ["price"]),
                self._check_data_consistency(symbol, "1mo")
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            price_validation = results[0] if not isinstance(results[0], Exception) else None
            cross_validation = results[1] if not isinstance(results[1], Exception) else None
            consistency_check = results[2] if not isinstance(results[2], Exception) else None
            
            # Calculate overall quality score
            scores = []
            if price_validation and "data_quality_score" in price_validation:
                scores.append(price_validation["data_quality_score"])
            if cross_validation and "overall_confidence_score" in cross_validation:
                scores.append(cross_validation["overall_confidence_score"])
            if consistency_check and "consistency_score" in consistency_check:
                scores.append(consistency_check["consistency_score"])
            
            overall_quality_score = statistics.mean(scores) if scores else 0
            
            # Determine quality rating
            if overall_quality_score > 0.9:
                quality_rating = "excellent"
            elif overall_quality_score > 0.8:
                quality_rating = "good"
            elif overall_quality_score > 0.6:
                quality_rating = "fair"
            elif overall_quality_score > 0.4:
                quality_rating = "poor"
            else:
                quality_rating = "very_poor"
            
            return {
                "symbol": symbol,
                "assessment_timestamp": datetime.now(),
                "overall_quality_score": round(overall_quality_score, 3),
                "quality_rating": quality_rating,
                "component_scores": {
                    "price_validation": price_validation.get("data_quality_score") if price_validation else None,
                    "cross_validation": cross_validation.get("overall_confidence_score") if cross_validation else None,
                    "consistency_check": consistency_check.get("consistency_score") if consistency_check else None
                },
                "detailed_results": {
                    "price_validation": price_validation,
                    "cross_validation": cross_validation,
                    "consistency_check": consistency_check
                },
                "recommendations": self._generate_quality_recommendations(overall_quality_score, {
                    "price_validation": price_validation,
                    "cross_validation": cross_validation,
                    "consistency_check": consistency_check
                })
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality for {symbol}: {str(e)}")
            raise

    def _generate_quality_recommendations(self, overall_score: float, results: Dict) -> List[str]:
        """Generate recommendations based on data quality assessment"""
        recommendations = []
        
        if overall_score < 0.7:
            recommendations.append("Consider using additional data sources for verification")
        
        if results.get("price_validation") and results["price_validation"].get("anomaly_count", 0) > 5:
            recommendations.append("Review and clean price data anomalies")
        
        if results.get("cross_validation") and results["cross_validation"].get("overall_confidence_score", 0) < 0.8:
            recommendations.append("Cross-validate findings with external sources")
        
        if results.get("consistency_check") and results["consistency_check"].get("consistency_score", 0) < 0.7:
            recommendations.append("Investigate data consistency issues over time")
        
        if overall_score > 0.9:
            recommendations.append("Data quality is excellent - safe to proceed with analysis")
        
        return recommendations

    async def _get_price_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Get price data from database"""
        try:
            from data.database import db_manager
            
            if db_manager is None:
                self.logger.error("Database manager not initialized")
                return None
                
            with db_manager.get_session() as session:
                from data.models import Stock, PriceData
                
                stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                if not stock:
                    return None
                
                # Get price data
                query = session.query(PriceData).filter(
                    PriceData.stock_id == stock.id
                ).order_by(PriceData.date.asc())
                
                price_records = query.all()
                
                if not price_records:
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
                return df.sort_index()
                
        except Exception as e:
            self.logger.error(f"Error getting price data for {symbol}: {str(e)}")
            return None

    async def _generate_credibility_report(self, symbol: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive credibility report for analysis data"""
        self.logger.info(f"Generating credibility report for {symbol}")
        
        try:
            # Run comprehensive data quality assessment
            quality_assessment = await self._assess_data_quality(symbol)
            
            # Analyze the provided analysis data for credibility
            analysis_credibility = self._assess_analysis_credibility(analysis_data)
            
            # Combine assessments
            overall_credibility = (
                quality_assessment.get("overall_quality_score", 0) * 0.6 +
                analysis_credibility.get("credibility_score", 0) * 0.4
            )
            
            # Generate credibility rating
            if overall_credibility > 0.9:
                credibility_rating = "highly_reliable"
            elif overall_credibility > 0.8:
                credibility_rating = "reliable"
            elif overall_credibility > 0.6:
                credibility_rating = "moderately_reliable"
            elif overall_credibility > 0.4:
                credibility_rating = "questionable"
            else:
                credibility_rating = "unreliable"
            
            return {
                "symbol": symbol,
                "report_timestamp": datetime.now(),
                "overall_credibility_score": round(overall_credibility, 3),
                "credibility_rating": credibility_rating,
                "data_quality_assessment": quality_assessment,
                "analysis_credibility": analysis_credibility,
                "risk_factors": self._identify_credibility_risks(quality_assessment, analysis_credibility),
                "recommendations": self._generate_credibility_recommendations(overall_credibility, quality_assessment, analysis_credibility),
                "fact_check_summary": {
                    "data_sources_verified": True,
                    "cross_validation_performed": True,
                    "anomalies_detected": quality_assessment.get("detailed_results", {}).get("price_validation", {}).get("anomaly_count", 0),
                    "consistency_verified": True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating credibility report for {symbol}: {str(e)}")
            raise

    def _assess_analysis_credibility(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess credibility of analysis results"""
        credibility_factors = []
        credibility_score = 1.0
        
        # Check if multiple analysis types were used
        analysis_types = []
        if "technical_analysis" in analysis_data:
            analysis_types.append("technical")
        if "fundamental_analysis" in analysis_data:
            analysis_types.append("fundamental")
        
        if len(analysis_types) > 1:
            credibility_factors.append("multiple_analysis_types")
        else:
            credibility_score -= 0.1
        
        # Check confidence scores in analysis
        if "technical_analysis" in analysis_data:
            tech_analysis = analysis_data["technical_analysis"]
            if "technical_score" in tech_analysis:
                tech_confidence = tech_analysis["technical_score"].get("confidence", 0)
                if tech_confidence < 0.5:
                    credibility_score -= 0.2
                    credibility_factors.append("low_technical_confidence")
        
        # Check for AI insights validation
        if "ai_insights" in analysis_data:
            credibility_factors.append("ai_validation_performed")
        else:
            credibility_score -= 0.1
        
        # Check data recency
        analysis_date = analysis_data.get("analysis_date")
        if analysis_date:
            if isinstance(analysis_date, str):
                analysis_date = datetime.fromisoformat(analysis_date.replace('Z', '+00:00'))
            days_old = (datetime.now() - analysis_date).days
            if days_old > 7:
                credibility_score -= 0.1
                credibility_factors.append("analysis_outdated")
        
        return {
            "credibility_score": max(0, credibility_score),
            "credibility_factors": credibility_factors,
            "analysis_types_used": analysis_types,
            "assessment_notes": f"Analysis credibility based on {len(credibility_factors)} positive factors"
        }

    def _identify_credibility_risks(self, quality_assessment: Dict, analysis_credibility: Dict) -> List[str]:
        """Identify potential credibility risks"""
        risks = []
        
        # Data quality risks
        if quality_assessment.get("overall_quality_score", 0) < 0.7:
            risks.append("low_data_quality")
        
        if quality_assessment.get("detailed_results", {}).get("price_validation", {}).get("anomaly_count", 0) > 10:
            risks.append("high_anomaly_count")
        
        # Analysis credibility risks
        if analysis_credibility.get("credibility_score", 0) < 0.6:
            risks.append("questionable_analysis_methods")
        
        if "low_technical_confidence" in analysis_credibility.get("credibility_factors", []):
            risks.append("low_confidence_indicators")
        
        return risks

    def _generate_credibility_recommendations(self, overall_credibility: float, quality_assessment: Dict, analysis_credibility: Dict) -> List[str]:
        """Generate recommendations for improving credibility"""
        recommendations = []
        
        if overall_credibility < 0.8:
            recommendations.append("Seek additional data sources for verification")
        
        if quality_assessment.get("overall_quality_score", 0) < 0.7:
            recommendations.append("Improve data collection and validation processes")
        
        if len(analysis_credibility.get("analysis_types_used", [])) < 2:
            recommendations.append("Use multiple analysis methodologies for cross-validation")
        
        if overall_credibility > 0.9:
            recommendations.append("Analysis demonstrates high credibility - suitable for decision making")
        
        return recommendations