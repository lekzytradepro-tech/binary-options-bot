import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from src.core.database import db

logger = logging.getLogger(__name__)

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    user = update.effective_user
    
    # Add user to database
    db.add_user(user.id, user.username, user.first_name)
    
    # Show main menu
    from src.bot.menus import show_main_menu
    await show_main_menu(update, context)

async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_text = """
📖 **Binary Options AI Pro - Help**

**Commands:**
/start - Start bot and show main menu
/help - Show this help message
/signals - Get quick AI signals

**AI Features:**
• 15 AI Engines for analysis
• Real-time market signals
• Multiple trading strategies
• Risk management

**Support:** Contact admin for assistance"""
    
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def handle_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /signals command"""
    from src.bot.menus import show_signals_menu
    await show_signals_menu(update, context)

async def handle_button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all button clicks"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == "menu_signals":
        from src.bot.menus import show_signals_menu
        await show_signals_menu(update, context)
    elif data == "menu_strategies":
        from src.bot.menus import show_strategies_menu
        await show_strategies_menu(update, context)
    elif data == "menu_account":
        from src.bot.menus import show_account_menu
        await show_account_menu(update, context)
    elif data.startswith("signal_"):
        from src.ai.signal_generator import generate_signal
        await generate_signal(update, context, data)
    else:
        await query.edit_message_text("🔄 Feature coming soon!")
