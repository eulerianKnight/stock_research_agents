import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import logging
from pathlib import Path


# Load environment variables from .env file
load_dotenv()

@dataclass
class APIConfig:
    """API configuration settings"""
    # Claude API
    anthropic_api_key: str = ""
    
    # Stock Market APIs
    alpha_vantage_api_key: str = ""
    yahoo_finance_enabled: bool = True
    
    # Rate limiting
    api_rate_limit: int = 60  # requests per minute
    api_timeout: int = 30  # seconds

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    database_url: str = ""
    max_connections: int = 5
    connection_timeout: int = 30

@dataclass
class AgentConfig:
    """Agent system configuration"""
    max_concurrent_tasks: int = 5
    task_timeout: int = 300  # 5 minutes
    retry_attempts: int = 3
    retry_delay: int = 5  # seconds

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/stock_research.log"

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent

        self.api = APIConfig()
        self.database = DatabaseConfig()
        self.agents = AgentConfig()
        self.logging = LoggingConfig()
        
        # Load configuration from environment variables
        self._load_from_env()
        
        # Setup logging
        self._setup_logging()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        
        # API Configuration
        self.api.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.api.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.api.yahoo_finance_enabled = os.getenv("YAHOO_FINANCE_ENABLED", "true").lower() == "true"
        self.api.api_rate_limit = int(os.getenv("API_RATE_LIMIT", "60"))
        self.api.api_timeout = int(os.getenv("API_TIMEOUT", "30"))
        
        # Database Configuration
        default_db_path = self.project_root / "data" / "stock_research.db"
        default_db_url = f"sqlite:///{default_db_path.resolve()}"
        # Ensure the directory exists
        os.makedirs(default_db_path.parent, exist_ok=True)

        self.database.database_url = os.getenv("DATABASE_URL", default_db_url)
        self.database.max_connections = int(os.getenv("DB_MAX_CONNECTIONS", "5"))
        self.database.connection_timeout = int(os.getenv("DB_CONNECTION_TIMEOUT", "30"))
        
        # Agent Configuration
        self.agents.max_concurrent_tasks = int(os.getenv("MAX_CONCURRENT_TASKS", "5"))
        self.agents.task_timeout = int(os.getenv("TASK_TIMEOUT", "300"))
        self.agents.retry_attempts = int(os.getenv("RETRY_ATTEMPTS", "3"))
        self.agents.retry_delay = int(os.getenv("RETRY_DELAY", "5"))
        
        # Logging Configuration
        self.logging.level = os.getenv("LOG_LEVEL", "INFO")
        self.logging.file_path = os.getenv("LOG_FILE_PATH", "logs/stock_research.log")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.logging.file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format,
            handlers=[
                logging.FileHandler(self.logging.file_path),
                logging.StreamHandler()  # Also log to console
            ]
        )
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check required API keys
        if not self.api.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required")
        
        # Validate numeric values
        if self.api.api_rate_limit <= 0:
            errors.append("API_RATE_LIMIT must be positive")
        
        if self.agents.max_concurrent_tasks <= 0:
            errors.append("MAX_CONCURRENT_TASKS must be positive")
        
        # Log errors
        if errors:
            logger = logging.getLogger("config")
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        return True
    
    def get_stock_symbols(self) -> list:
        """Get list of Indian stock symbols to track"""
        # Default list of popular Indian stocks
        default_symbols = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", 
            "HINDUNILVR.NS", "ICICIBANK.NS", "KOTAKBANK.NS", 
            "LT.NS", "SBIN.NS", "BHARTIARTL.NS"
        ]
        
        # Can be overridden via environment variable
        # env_symbols = os.getenv("STOCK_SYMBOLS", "")
        # if env_symbols:
        #     return [symbol.strip() for symbol in env_symbols.split(",")]
        
        return default_symbols
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for debugging)"""
        return {
            "api": {
                "anthropic_api_key": "***" if self.api.anthropic_api_key else "",
                "alpha_vantage_api_key": "***" if self.api.alpha_vantage_api_key else "",
                "yahoo_finance_enabled": self.api.yahoo_finance_enabled,
                "api_rate_limit": self.api.api_rate_limit,
                "api_timeout": self.api.api_timeout
            },
            "database": {
                "database_url": self.database.database_url,
                "max_connections": self.database.max_connections,
                "connection_timeout": self.database.connection_timeout
            },
            "agents": {
                "max_concurrent_tasks": self.agents.max_concurrent_tasks,
                "task_timeout": self.agents.task_timeout,
                "retry_attempts": self.agents.retry_attempts,
                "retry_delay": self.agents.retry_delay
            },
            "logging": {
                "level": self.logging.level,
                "file_path": self.logging.file_path
            }
        }

# Global configuration instance
config = Config()