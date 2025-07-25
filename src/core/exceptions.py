class MagicbotError(Exception):
    """Base exception for Magicbot trading system"""
    pass

class ConfigurationError(MagicbotError):
    """Configuration related errors"""
    pass

class ExchangeError(MagicbotError):
    """Exchange integration errors"""
    pass

class StrategyError(MagicbotError):
    """Strategy execution errors"""
    pass

class RiskManagementError(MagicbotError):
    """Risk management errors"""
    pass

class DataError(MagicbotError):
    """Data processing errors"""
    pass