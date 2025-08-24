"""
Phase 3B Configuration Settings
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8004  # Different from Phase 3A (8003)
    DEBUG: bool = True
    
    # Phase 3A Integration
    PHASE3A_BASE_URL: str = "http://127.0.0.1:8003"
    PHASE3A_TIMEOUT: int = 30
    
    # Resource Configuration
    MAX_COMPUTE_RESOURCES: float = 100.0
    MAX_STORAGE_RESOURCES: float = 1000.0
    MAX_NETWORK_RESOURCES: float = 50.0
    MAX_MODEL_SLOTS: int = 10
    
    # Scheduling Configuration
    SCHEDULER_INTERVAL_SECONDS: int = 5
    MONITOR_INTERVAL_SECONDS: int = 10
    MAX_CONCURRENT_EXPERIMENTS: int = 5
    
    # Database Configuration (optional)
    DATABASE_URL: Optional[str] = None
    USE_DATABASE: bool = False
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security Configuration
    CORS_ORIGINS: List[str] = ["*"]
    API_KEY_REQUIRED: bool = False
    API_KEYS: List[str] = []
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9091
    
    # Experiment Configuration
    DEFAULT_EXPERIMENT_TIMEOUT_HOURS: int = 24
    MAX_EXPERIMENT_DURATION_HOURS: int = 168  # 1 week
    AUTO_CLEANUP_COMPLETED_EXPERIMENTS: bool = True
    CLEANUP_AFTER_HOURS: int = 48
    
    # Integration Settings
    ENABLE_SLACK_NOTIFICATIONS: bool = False
    SLACK_WEBHOOK_URL: Optional[str] = None
    
    ENABLE_EMAIL_NOTIFICATIONS: bool = False
    SMTP_SERVER: Optional[str] = None
    SMTP_PORT: int = 587
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Environment-specific configurations
def get_development_settings():
    """Development environment settings"""
    return Settings(
        DEBUG=True,
        HOST="127.0.0.1",
        PORT=8004,
        LOG_LEVEL="DEBUG"
    )

def get_production_settings():
    """Production environment settings"""
    return Settings(
        DEBUG=False,
        HOST="0.0.0.0",
        PORT=8004,
        LOG_LEVEL="INFO",
        CORS_ORIGINS=["https://yourdomain.com"],
        API_KEY_REQUIRED=True
    )

def get_test_settings():
    """Test environment settings"""
    return Settings(
        DEBUG=True,
        HOST="127.0.0.1",
        PORT=8005,  # Different port for testing
        LOG_LEVEL="DEBUG",
        USE_DATABASE=False
    )