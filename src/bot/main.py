import logging
import asyncio
from telegram.ext import Application, CommandHandler

logger = logging.getLogger(__name__)

class BinaryOptionsBot:
    def __init__(self):
        self.application = None
    
    async def setup(self):
        """Setup bot application"""
        from src.core.config import Config
        from src.core.database import db
        
        # Create application
        self.application = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        
        # Add core handlers
        self.application.add_handler(CommandHandler("start", self.handle_start))
        self.application.add_handler(CommandHandler("help", self.handle_help))
        
        logger.info("Bot setup completed")
        return self.application
    
    async def handle_start(self, update, context):
        """Handle /start command"""
        from src.core.database import db
        
        user = update.effective_user
        db.add_user(user.id, user.username, user.first_name)
        
        welcome_text = """
🤖 Welcome to Binary Options AI Pro!

I'm your AI-powered trading assistant.

🚀 Getting Started:
• Use /help to see commands
• More features coming soon!

Stay tuned for AI signals!"""
        
        await update.message.reply_text(welcome_text)
        logger.info(f"New user: {user.id}")
    
    async def handle_help(self, update, context):
        """Handle /help command"""
        help_text = """
📖 Available Commands:

/start - Start the bot
/help - Show this message

💡 More features coming soon!"""
        
        await update.message.reply_text(help_text)
    
    async def run(self):
        """Run the bot with proper shutdown handling"""
        app = await self.setup()
        
        try:
            logger.info("Starting bot polling...")
            await app.initialize()
            await app.start()
            await app.updater.start_polling()
            
            # Keep the bot running
            await asyncio.Event().wait()
            
        except KeyboardInterrupt:
            logger.info("Bot stopping...")
        finally:
            # Proper shutdown
            if app.updater:
                await app.updater.stop()
            await app.stop()
            await app.shutdown()

# Fixed main function
async def main():
    bot = BinaryOptionsBot()
    await bot.run()
