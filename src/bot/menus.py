from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show main navigation menu"""
    keyboard = [
        [InlineKeyboardButton("🚀 Get AI Signals", callback_data="menu_signals")],
        [InlineKeyboardButton("📊 Trading Strategies", callback_data="menu_strategies")],
        [InlineKeyboardButton("💼 Account Dashboard", callback_data="menu_account")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = """
🤖 **Binary Options AI Pro** 🚀

*Your AI Trading Assistant*

🎯 **Powered by 15 AI Engines**
💎 **Account:** FREE TRIAL
📊 **Signals Today:** 0/3 used

*Tap buttons to explore features!*"""
    
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    else:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")

async def show_signals_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show signals menu"""
    keyboard = [
        [InlineKeyboardButton("⚡ Quick Signal (1-5min)", callback_data="signal_quick")],
        [InlineKeyboardButton("📈 Trend Analysis", callback_data="signal_trend")],
        [InlineKeyboardButton("🎯 Pattern Scanner", callback_data="signal_pattern")],
        [InlineKeyboardButton("🔙 Main Menu", callback_data="menu_main")]
    ]
    
    text = "📊 **AI Trading Signals**\n\nChoose your signal type:"
    
    await update.callback_query.edit_message_text(
        text, 
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )

async def show_strategies_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show strategies menu"""
    keyboard = [
        [InlineKeyboardButton("🚀 Trend Spotter", callback_data="strategy_trend")],
        [InlineKeyboardButton("⚡ Scalper Pro", callback_data="strategy_scalper")],
        [InlineKeyboardButton("📊 Volume Analysis", callback_data="strategy_volume")],
        [InlineKeyboardButton("🔙 Main Menu", callback_data="menu_main")]
    ]
    
    text = "🎯 **Trading Strategies**\n\nChoose your trading style:"
    
    await update.callback_query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )

async def show_account_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show account menu"""
    from src.core.database import db
    
    user = update.effective_user
    user_data = db.get_user(user.id)
    
    keyboard = [
        [InlineKeyboardButton("💎 Upgrade Plan", callback_data="account_upgrade")],
        [InlineKeyboardButton("📊 Usage Stats", callback_data="account_stats")],
        [InlineKeyboardButton("🔙 Main Menu", callback_data="menu_main")]
    ]
    
    text = f"""
💼 **Account Management**

👤 **User:** {user.first_name}
🆔 **ID:** {user.id}
📅 **Status:** ✅ Active

**Plan:** 🆓 Free Trial
**Signals Used:** 0/3 today
**AI Engines:** 15 Available

💎 **Upgrade for unlimited signals!**"""
    
    await update.callback_query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )
