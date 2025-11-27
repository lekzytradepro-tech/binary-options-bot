import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from src.core.database import db
from src.core.config import Config

logger = logging.getLogger(__name__)

def get_user_info(update: Update):
    """Safely extract user information from update"""
    try:
        if update.effective_user:
            return {
                'id': update.effective_user.id,
                'username': update.effective_user.username,
                'first_name': update.effective_user.first_name
            }
        elif update.message and update.message.from_user:
            return {
                'id': update.message.from_user.id,
                'username': update.message.from_user.username,
                'first_name': update.message.from_user.first_name
            }
        elif update.callback_query and update.callback_query.from_user:
            return {
                'id': update.callback_query.from_user.id,
                'username': update.callback_query.from_user.username,
                'first_name': update.callback_query.from_user.first_name
            }
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
    
    return {'id': 0, 'username': 'unknown', 'first_name': 'User'}

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command for binary options"""
    try:
        user_info = get_user_info(update)
        
        # Add user to database
        try:
            db.add_user(user_info['id'], user_info['username'], user_info['first_name'])
        except Exception as e:
            logger.warning(f"Could not add user to database: {e}")
        
        # Show legal disclaimer first
        from src.bot.menus import show_legal_disclaimer
        await show_legal_disclaimer(update, context)
        
        logger.info(f"👤 User started: {user_info['id']} - {user_info['first_name']}")
        
    except Exception as e:
        logger.error(f"Error in handle_start: {e}")
        # Simple fallback response
        try:
            if update.message:
                await update.message.reply_text(
                    "🤖 Welcome to Binary Options AI Pro! 🚀\n\n"
                    "Use /help for assistance.",
                    parse_mode="Markdown"
                )
        except Exception as reply_error:
            logger.error(f"Fallback response failed: {reply_error}")

async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_text = """
📖 **Binary Options AI Pro - Help**

**Commands:**
/start - Start bot
/help - Show help  
/signals - Get signals
/assets - View assets
/status - Check status

**Features:**
• AI-powered signals
• 15 trading assets
• Risk management
• Educational content

*Use menu buttons for best experience*"""
    
    try:
        if update.message:
            await update.message.reply_text(help_text, parse_mode="Markdown")
        elif update.callback_query:
            await update.callback_query.message.reply_text(help_text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in handle_help: {e}")

async def handle_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /signals command"""
    try:
        from src.ai.signal_generator import handle_quick_signal
        await handle_quick_signal(update, context)
    except Exception as e:
        logger.error(f"Error in handle_signals: {e}")
        try:
            if update.message:
                await update.message.reply_text(
                    "🎯 **Signals**\n\nUse the menu buttons to access signals.",
                    parse_mode="Markdown"
                )
        except Exception as reply_error:
            logger.error(f"Signals fallback failed: {reply_error}")

async def handle_assets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show available assets"""
    try:
        keyboard = []
        
        # Use a subset of assets for reliability
        assets = ["EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD", "XAU/USD"]
        
        for asset in assets:
            keyboard.append([InlineKeyboardButton(f"💱 {asset}", callback_data=f"asset_{asset}")])
        
        keyboard.append([InlineKeyboardButton("🔙 Main Menu", callback_data="menu_main")])
        
        text = "📊 **Available Assets**\n\nSelect an asset to trade:"
        
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
    except Exception as e:
        logger.error(f"Error in handle_assets: {e}")

async def handle_button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks with robust error handling"""
    try:
        query = update.callback_query
        await query.answer()
        
        data = query.data
        logger.info(f"Button: {data}")
        
        # Handle different button types
        if data == "disclaimer_accepted":
            from src.bot.menus import show_binary_main_menu
            await show_binary_main_menu(update, context)
            
        elif data == "disclaimer_declined":
            await query.edit_message_text("❌ Disclaimer declined. Use /start to try again.")
            
        elif data == "menu_main":
            from src.bot.menus import show_binary_main_menu
            await show_binary_main_menu(update, context)
            
        elif data == "menu_signals":
            from src.bot.menus import show_signals_menu
            await show_signals_menu(update, context)
            
        elif data == "menu_assets":
            await handle_assets(update, context)
            
        elif data.startswith("asset_"):
            asset = data.replace("asset_", "")
            from src.bot.menus import show_asset_expiry_menu
            await show_asset_expiry_menu(update, context, asset)
            
        elif data.startswith("signal_") or data.startswith("expiry_"):
            from src.ai.signal_generator import generate_binary_signal
            await generate_binary_signal(update, context, data)
            
        elif data.startswith("strategy_"):
            from src.bot.menus import show_strategy_info
            await show_strategy_info(update, context, data)
            
        elif data.startswith("menu_"):
            # Handle other menu items
            await query.edit_message_text(
                "🔄 **Loading...**\n\nPlease wait while we load this feature.",
                parse_mode="Markdown"
            )
            
        else:
            await query.edit_message_text(
                "❌ **Feature Not Available**\n\nThis feature is currently unavailable.",
                parse_mode="Markdown"
            )
            
    except Exception as e:
        logger.error(f"Button handler error: {e}")
        try:
            await update.callback_query.edit_message_text(
                "❌ **Error**\n\nPlease try again or use /start.",
                parse_mode="Markdown"
            )
        except:
            pass

async def handle_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle unknown commands"""
    try:
        if update.message:
            await update.message.reply_text(
                "🤖 I didn't understand that. Use /help for commands.",
                parse_mode="Markdown"
            )
    except Exception as e:
        logger.error(f"Error in handle_unknown: {e}")

async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    try:
        status_text = "✅ **Bot Status: Operational**\n\nAll systems are working normally."
        if update.message:
            await update.message.reply_text(status_text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in handle_status: {e}")

async def handle_quick_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /quickstart command"""
    try:
        quickstart_text = "🚀 **Quick Start**\n\nUse /start to begin with the main menu."
        if update.message:
            await update.message.reply_text(quickstart_text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in handle_quick_start: {e}")
