import logging
import asyncio
from telegram.ext import Application, CommandHandler

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update, context):
    """Handle /start command"""
    from src.core.database import db
    
    user = update.effective_user
    db.add_user(user.id, user.username, user.first_name)
    
    welcome_text = "🤖 Welcome to Binary Options AI Pro!"
    await update.message.reply_text(welcome_text)
    logger.info(f"New user: {user.id}")

async def help(update, context):
    """Handle /help command"""
    await update.message.reply_text("📖 Use /start to begin")

async def main():
    """Main async function - SIMPLIFIED"""
    from src.core.config import Config
    
    # Create bot application
    application = Application.builder().token(Config.TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help))
    
    # Run bot
    logger.info("Starting bot polling...")
    await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
