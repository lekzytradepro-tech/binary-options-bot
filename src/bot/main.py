import logging
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logger = logging.getLogger(__name__)

class BinaryOptionsBot:
    def __init__(self):
        self.application = None
    
    async def setup(self):
        from src.core.config import Config
        from src.core.database import db
        
        # Create application
        self.application = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        
        # Add handlers
        self.application.add_handler(CommandHandler("start", self.handle_start))
        self.application.add_handler(CallbackQueryHandler(self.handle_button_click))
        
        # Initialize application
        await self.application.initialize()
        await self.application.start()
        
        logger.info("✅ Bot setup completed successfully")
    
    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from src.core.database import db
        
        user = update.effective_user
        db.add_user(user.id, user.username, user.first_name)
        
        # Send main menu with buttons
        await self.show_main_menu(update, context)
    
    async def show_main_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show main navigation menu with buttons"""
        keyboard = [
            [InlineKeyboardButton("🚀 Get Signals", callback_data="menu_signals")],
            [InlineKeyboardButton("📊 Trading Strategies", callback_data="menu_strategies")],
            [InlineKeyboardButton("💼 Account", callback_data="menu_account")],
            [InlineKeyboardButton("📚 Education", callback_data="menu_education")],
            [InlineKeyboardButton("🛠️ Admin", callback_data="menu_admin")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        text = """
🤖 **Binary Options AI Pro** - Professional Trading Platform

*Navigate using buttons below:*

🎯 **Get Signals** - AI-powered trading signals
📊 **Strategies** - Choose your trading style  
💼 **Account** - Manage your subscription
📚 **Education** - Learn trading strategies
🛠️ **Admin** - System controls

*No commands needed - just click buttons!*"""
        
        if update.callback_query:
            await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")
        else:
            await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    
    async def handle_button_click(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle all button clicks"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "menu_signals":
            await self.show_signals_menu(update, context)
        elif data == "menu_strategies":
            await self.show_strategies_menu(update, context)
        elif data == "menu_account":
            await self.show_account_menu(update, context)
        elif data == "menu_education":
            await self.show_education_menu(update, context)
        elif data == "menu_admin":
            await self.show_admin_menu(update, context)
        elif data == "back_main":
            await self.show_main_menu(update, context)
        else:
            await query.edit_message_text("🔄 Feature coming soon!")
    
    async def show_signals_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show signals menu"""
        keyboard = [
            [InlineKeyboardButton("⚡ Quick Signal", callback_data="signal_quick")],
            [InlineKeyboardButton("📈 Trend Analysis", callback_data="signal_trend")],
            [InlineKeyboardButton("🎯 Pattern Scanner", callback_data="signal_pattern")],
            [InlineKeyboardButton("🔙 Back", callback_data="back_main")]
        ]
        
        text = """
📊 **Trading Signals Menu**

Choose your signal type:

⚡ **Quick Signal** - Fast 1-5 minute signals
📈 **Trend Analysis** - Detailed trend-based signals  
🎯 **Pattern Scanner** - AI pattern recognition

*Professional AI analysis in seconds*"""
        
        await update.callback_query.edit_message_text(
            text, 
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    
    async def show_strategies_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show strategies menu"""
        keyboard = [
            [InlineKeyboardButton("🚀 Trend Spotter", callback_data="strategy_trend")],
            [InlineKeyboardButton("⚡ Scalper Pro", callback_data="strategy_scalper")],
            [InlineKeyboardButton("📊 Volume Analysis", callback_data="strategy_volume")],
            [InlineKeyboardButton("🔙 Back", callback_data="back_main")]
        ]
        
        text = """
🎯 **Trading Strategies**

Choose your trading style:

🚀 **Trend Spotter** - Follow market trends
⚡ **Scalper Pro** - Quick 1-3 minute trades
📊 **Volume Analysis** - Volume-based signals

*Each strategy uses different AI engines*"""
        
        await update.callback_query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    
    async def show_account_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show account management menu"""
        from src.core.database import db
        
        user = update.effective_user
        user_data = db.get_user(user.id)
        
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade Plan", callback_data="account_upgrade")],
            [InlineKeyboardButton("📊 Usage Stats", callback_data="account_stats")],
            [InlineKeyboardButton("🆓 Free Trial", callback_data="account_trial")],
            [InlineKeyboardButton("🔙 Back", callback_data="back_main")]
        ]
        
        text = f"""
💼 **Account Management**

👤 **User:** {user.first_name}
🆔 **ID:** {user.id}
📅 **Member since:** {user_data['created_at'] if user_data else 'Recently'}

**Plan:** 🆓 Free Trial
**Signals today:** 0/3
**Status:** ✅ Active

*Upgrade for unlimited signals*"""
        
        await update.callback_query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    
    async def show_education_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show education menu"""
        keyboard = [
            [InlineKeyboardButton("📚 Trading Basics", callback_data="edu_basics")],
            [InlineKeyboardButton("🎯 Risk Management", callback_data="edu_risk")],
            [InlineKeyboardButton("🤖 AI Strategies", callback_data="edu_ai")],
            [InlineKeyboardButton("🔙 Back", callback_data="back_main")]
        ]
        
        text = """
📚 **Trading Education Center**

Learn professional trading:

📚 **Trading Basics** - Fundamental concepts
🎯 **Risk Management** - Protect your capital
🤖 **AI Strategies** - How our AI works

*Knowledge is power in trading*"""
        
        await update.callback_query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    
    async def show_admin_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show admin menu (only for admins)"""
        from src.core.config import Config
        
        user = update.effective_user
        
        if user.id not in Config.ADMIN_IDS:
            await update.callback_query.edit_message_text("❌ Admin access required")
            return
        
        keyboard = [
            [InlineKeyboardButton("📊 System Stats", callback_data="admin_stats")],
            [InlineKeyboardButton("👥 User Management", callback_data="admin_users")],
            [InlineKeyboardButton("🔧 Bot Controls", callback_data="admin_controls")],
            [InlineKeyboardButton("🔙 Back", callback_data="back_main")]
        ]
        
        text = """
🛠️ **Admin Panel**

System administration tools:

📊 **System Stats** - Bot performance metrics
👥 **User Management** - Manage users and plans  
🔧 **Bot Controls** - System configuration

*Restricted access*"""
        
        await update.callback_query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    
    async def run_polling(self):
        """Run bot with polling - for development"""
        await self.application.run_polling()
    
    async def cleanup(self):
        """Cleanup bot resources"""
        if self.application:
            await self.application.stop()
            await self.application.shutdown()

# Render-compatible main function
async def main():
    """Main async function for Render"""
    bot = BinaryOptionsBot()
    try:
        await bot.setup()
        logger.info("🤖 Bot started successfully - waiting for messages...")
        
        # Keep the bot running
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        await bot.cleanup()

# Development entry point
def run_bot():
    """Run bot for development"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Failed to run bot: {e}")

if __name__ == "__main__":
    run_bot()
