import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logger = logging.getLogger(__name__)

def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    from src.core.database import db
    
    user = update.effective_user
    db.add_user(user.id, user.username, user.first_name)
    
    # Show main menu
    keyboard = [
        [InlineKeyboardButton("🚀 Get Signals", callback_data="menu_signals")],
        [InlineKeyboardButton("📊 Trading Strategies", callback_data="menu_strategies")],
        [InlineKeyboardButton("💼 Account", callback_data="menu_account")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = """
🤖 **Binary Options AI Pro**

*Professional Trading Platform*

🎯 Get AI-powered trading signals
📊 Multiple trading strategies  
💼 Manage your account

*Click buttons to navigate*"""
    
    update.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    logger.info(f"User {user.id} started bot")

def handle_button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks"""
    query = update.callback_query
    query.answer()
    
    data = query.data
    
    if data == "menu_signals":
        keyboard = [
            [InlineKeyboardButton("⚡ Quick Signal", callback_data="signal_quick")],
            [InlineKeyboardButton("📈 Trend Analysis", callback_data="signal_trend")],
            [InlineKeyboardButton("🔙 Back", callback_data="back_main")]
        ]
        query.edit_message_text(
            "📊 **Trading Signals** - Choose signal type",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    elif data == "menu_strategies":
        keyboard = [
            [InlineKeyboardButton("🚀 Trend Spotter", callback_data="strategy_trend")],
            [InlineKeyboardButton("⚡ Scalper Pro", callback_data="strategy_scalper")],
            [InlineKeyboardButton("🔙 Back", callback_data="back_main")]
        ]
        query.edit_message_text(
            "🎯 **Trading Strategies** - Choose your style",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    elif data == "menu_account":
        from src.core.database import db
        user = update.effective_user
        user_data = db.get_user(user.id)
        
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade Plan", callback_data="account_upgrade")],
            [InlineKeyboardButton("📊 Usage Stats", callback_data="account_stats")],
            [InlineKeyboardButton("🔙 Back", callback_data="back_main")]
        ]
        
        text = f"""
💼 **Account Management**

👤 **User:** {user.first_name}
🆔 **ID:** {user.id}
📅 **Status:** ✅ Active

**Plan:** 🆓 Free Trial"""
        
        query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    elif data == "back_main":
        keyboard = [
            [InlineKeyboardButton("🚀 Get Signals", callback_data="menu_signals")],
            [InlineKeyboardButton("📊 Trading Strategies", callback_data="menu_strategies")],
            [InlineKeyboardButton("💼 Account", callback_data="menu_account")],
        ]
        query.edit_message_text(
            "🤖 **Binary Options AI Pro** - Main Menu",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    else:
        query.edit_message_text("🔄 Feature coming soon!")

def run_bot():
    """Run the bot - SIMPLE & RELIABLE"""
    from src.core.config import Config
    
    try:
        # Create application
        application = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", handle_start))
        application.add_handler(CommandHandler("help", handle_start))
        application.add_handler(CallbackQueryHandler(handle_button_click))
        
        logger.info("✅ Bot setup completed")
        logger.info("🤖 Starting bot polling...")
        
        # Start polling
        application.run_polling(
            drop_pending_updates=True,
            allowed_updates=['message', 'callback_query']
        )
        
    except Exception as e:
        logger.error(f"❌ Bot failed: {e}")
        raise

if __name__ == "__main__":
    run_bot()
