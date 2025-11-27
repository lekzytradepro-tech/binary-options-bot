import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Core configuration"""
    
    # Essential settings
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/bot.db")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Admin IDs
    ADMIN_IDS = []
    admin_env = os.getenv("ADMIN_IDS", "")
    if admin_env:
        try:
            ADMIN_IDS = [int(id.strip()) for id in admin_env.split(",")]
        except ValueError:
            logger.warning("Invalid ADMIN_IDS format")
    
    # Feature flags
    ENABLE_AI = os.getenv("ENABLE_AI", "false").lower() == "true"
    ENABLE_PAYMENTS = os.getenv("ENABLE_PAYMENTS", "false").lower() == "true"
    
    @classmethod
    def validate(cls):
        if not cls.TELEGRAM_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        logger.info("Configuration validated")
        return True

Config.validate()
