import logging
from telegram.ext import Application, CommandHandler

logger = logging.getLogger(__name__)

class BinaryOptionsBot:
    def __init__(self):
        self.application = None
    
    async def setup(self):
        from src.core.config import Config
        from src.core.database import db
        
        self.application = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        
        # Add core handlers
        self.application.add_handler(CommandHandler("start", self.handle_start))
        self.application.add_handler(CommandHandler("help", self.handle_help))
        
        logger.info("Bot setup completed")
    
    async def handle_start(self, update, context):
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
        help_text = """
📖 Available Commands:

/start - Start the bot
/help - Show this message

💡 More features coming soon!"""
        
        await update.message.reply_text(help_text)
    
    async def run(self):
        await self.setup()
        logger.info("Starting bot polling...")
        await self.application.run_polling()

async def main():
    bot = BinaryOptionsBot()
    await bot.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
