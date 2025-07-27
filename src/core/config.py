from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings
from typing import List, Dict, Any
import yaml
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DatabaseConfig(BaseModel):
    url: str
    pool_size: int = 10
    max_overflow: int = 20

class RedisConfig(BaseModel):
    url: str
    decode_responses: bool = True

class ExchangeConfig(BaseModel):
    testnet: bool = True
    api_key: str
    secret_key: str

class RiskManagementConfig(BaseModel):
    max_position_size: float = 0.1
    max_daily_loss: float = 0.05
    stop_loss_percentage: float = 0.02

class Settings(BaseSettings):
    app_name: str = "Magicbot"
    version: str = "0.1.0"
    debug: bool = True
    log_level: str = "INFO"
    environment: str = "development"
    
    # Database & Cache
    database_url: str
    redis_url: str
    
    # Exchange
    binance_api_key: str
    binance_secret_key: str
    binance_testnet: bool = True
    
    model_config = ConfigDict(
        env_file=".env",
        extra="allow"
    )

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file"""
    env = os.getenv("ENVIRONMENT", "development")
    config_path = f"config/{env}.yaml"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Expand environment variables
    config = _expand_env_vars(config)
    return config

def _expand_env_vars(obj):
    """Recursively expand environment variables in config"""
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        env_var = obj[2:-1]
        return os.getenv(env_var, obj)
    return obj

# Global settings instance
settings = Settings()
config = load_config()