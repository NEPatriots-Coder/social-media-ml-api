import os
from pydantic import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # Application settings
    app_name: str = "Social Media ML API"
    debug: bool = False
    environment: str = "development"
    
    # API settings
    api_v1_prefix: str = "/v1"
    
    # CORS settings
    allowed_origins: List[str] = ["*"]
    allowed_hosts: List[str] = ["*"]
    
    # Model settings
    models_path: str = "/app/data/models"
    
    # Redis settings (for caching predictions)
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600  # 1 hour
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # AWS settings
    aws_region: str = "us-east-1"
    s3_bucket: str = ""
    
    # Security settings
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings