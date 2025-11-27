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

💱 **Forex Majors** - High liquidity, tight spreads
₿ **Cryptocurrencies** - High volatility, 24/7 trading  
🟡 **Commodities** - Gold & Silver, safe havens
📈 **Indices** - Market indices, session based

*Click any asset to get signals*"""
    
    if update.callback_query:
        await update.callback_query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

async def handle_button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all binary options button clicks"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    try:
        # Main menu navigation
        if data == "menu_signals":
            from src.bot.menus import show_signals_menu
            await show_signals_menu(update, context)
            
        elif data == "menu_strategies":
            from src.bot.menus import show_strategies_menu
            await show_strategies_menu(update, context)
            
        elif data == "menu_account":
            from src.bot.menus import show_account_menu
            await show_account_menu(update, context)
            
        elif data == "menu_education":
            from src.bot.menus import show_education_menu
            await show_education_menu(update, context)
            
        elif data == "menu_main":
            from src.bot.menus import show_binary_main_menu
            await show_binary_main_menu(update, context)
        
        # Signal types
        elif data.startswith("signal_"):
            from src.ai.signal_generator import generate_binary_signal
            await generate_binary_signal(update, context, data)
            
        # Asset selection
        elif data.startswith("asset_"):
            asset = data.replace("asset_", "")
            from src.bot.menus import show_asset_expiry_menu
            await show_asset_expiry_menu(update, context, asset)
            
        # Expiry selection
        elif data.startswith("expiry_"):
            parts = data.split("_")
            if len(parts) >= 3:
                asset = parts[1]
                expiry = parts[2]
                from src.ai.signal_generator import generate_binary_signal
                await generate_binary_signal(update, context, f"signal_{asset}_{expiry}")
        
        # Strategy types  
        elif data.startswith("strategy_"):
            from src.bot.menus import show_strategy_info
            await show_strategy_info(update, context, data)
            
        else:
            await query.edit_message_text(
                "🔄 **Feature Coming Soon**\n\n*This binary options feature is in development.*",
                parse_mode="Markdown"
            )
            
    except Exception as e:
        logger.error(f"Button handler error: {e}")
        await query.edit_message_text(
            "❌ **Error Processing Request**\n\n*Please try again or contact support.*",
            parse_mode="Markdown"
        )

async def handle_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle unknown commands"""
    await update.message.reply_text(
        "🤖 **Binary Options AI Pro**\n\n"
        "I didn't understand that command. Here are available commands:\n\n"
        "• /start - Start binary trading\n"
        "• /help - Get help with binary options\n"  
        "• /signals - Get AI trading signals\n"
        "• /assets - View trading assets\n\n"
        "*Use buttons for the best experience!*",
        parse_mode="Markdown"
            )
