import logging
import asyncio
from telegram.ext import Application, CommandHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def start(update, context):
    from src.core.database import db
    
    user = update.effective_user
    db.add_user(user.id, user.username, user.first_name)
    
    welcome_text = "🤖 Welcome to Binary Options AI Pro!"
    await update.message.reply_text(welcome_text)
    logger.info(f"New user: {user.id}")

async def help(update, context):
    await update.message.reply_text("📖 Use /start to begin")

async def run_bot():
    """Run bot with proper cleanup"""
    from src.core.config import Config
    
    # Create bot application
    application = Application.builder().token(Config.TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help))
    
    # Start bot with proper cleanup
    logger.info("🤖 Telegram bot starting...")
    
    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        
        # Keep running until stopped
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Bot stopping...")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise
    finally:
        # Proper cleanup
        if application.updater:
            await application.updater.stop()
        await application.stop()
        await application.shutdown()

def main():
    """Main entry point for bot"""
    asyncio.run(run_bot())

if __name__ == "__main__":
    main()
