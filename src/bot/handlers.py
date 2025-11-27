import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from src.core.database import db
from src.core.config import Config

logger = logging.getLogger(__name__)

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command for binary options"""
    user = update.effective_user
    
    # Add user to database
    db.add_user(user.id, user.username, user.first_name)
    
    # Show binary options main menu
    from src.bot.menus import show_binary_main_menu
    await show_binary_main_menu(update, context)
    
    logger.info(f"👤 New user started: {user.id} - {user.first_name}")

async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command with binary options focus"""
    help_text = """
📖 **Binary Options AI Pro - Help**

🤖 *AI-Powered Binary Options Trading*

**Commands:**
/start - Start bot with binary options
/help - Show this help message  
/signals - Get quick binary signals
/assets - View available assets

**Binary Options Features:**
• 🎯 **CALL/PUT Signals** - AI direction predictions
• ⏰ **Smart Expiry** - 1min to 60min timeframes
• 📊 **Real Market Data** - Live TwelveData feeds
• 🤖 **8 AI Engines** - Specialized for binary trading
• 💰 **Payout Calculations** - Based on volatility
• 📈 **15 Assets** - Forex, Crypto, Commodities

**Risk Management:**
• Start with demo account
• Risk only 1-2% per trade
• Use stop losses
• Trade during active sessions

*Binary options trading involves high risk*"""
    
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def handle_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /signals command - quick binary signals"""
    from src.ai.signal_generator import handle_quick_signal
    await handle_quick_signal(update, context)

async def handle_assets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show available binary trading assets"""
    keyboard = []
    
    # Group assets by category
    forex_assets = [p for p in Config.BINARY_PAIRS if '/' in p and 'XAU' not in p and 'BTC' not in p]
    crypto_assets = [p for p in Config.BINARY_PAIRS if 'BTC' in p or 'ETH' in p]
    commodity_assets = [p for p in Config.BINARY_PAIRS if 'XAU' in p or 'XAG' in p]
    indices_assets = [p for p in Config.BINARY_PAIRS if 'US30' in p]
    
    # Add forex assets
    for i in range(0, len(forex_assets), 2):
        row = []
        for asset in forex_assets[i:i+2]:
            row.append(InlineKeyboardButton(f"💱 {asset}", callback_data=f"asset_{asset}"))
        keyboard.append(row)
    
    # Add crypto assets
    for asset in crypto_assets:
        keyboard.append([InlineKeyboardButton(f"₿ {asset}", callback_data=f"asset_{asset}")])
    
    # Add commodities
    for asset in commodity_assets:
        keyboard.append([InlineKeyboardButton(f"🟡 {asset}", callback_data=f"asset_{asset}")])
    
    # Add indices
    for asset in indices_assets:
        keyboard.append([InlineKeyboardButton(f"📈 {asset}", callback_data=f"asset_{asset}")])
    
    keyboard.append([InlineKeyboardButton("🔙 Main Menu", callback_data="menu_main")])
    
    text = """
📊 **Available Binary Trading Assets**

*Trade these assets with AI signals:*

💱 **Forex Majors** - High liquidity
