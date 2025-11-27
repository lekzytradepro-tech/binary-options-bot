import logging
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from src.api.twelvedata_client import twelvedata_client
from src.core.database import db

logger = logging.getLogger(__name__)

async def generate_binary_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, signal_type: str):
    """Generate binary options signal with real data"""
    query = update.callback_query
    
    # Extract symbol and expiry from signal_type (e.g., "signal_EURUSD_5")
    parts = signal_type.split('_')
    symbol = "EUR/USD"  # Default
    expiry = 5  # Default 5 minutes
    
    if len(parts) >= 3:
        symbol = parts[1].replace('USD', '/USD')  # Convert EURUSD to EUR/USD
        try:
            expiry = int(parts[2])
        except:
            expiry = 5
    
    # Show processing message
    processing_text = f"""
🔄 **AI Binary Analysis** 🤖

*Analyzing {symbol} for {expiry}-minute binary option...*

📊 Fetching real market data...
🤖 Running 8 AI engines...
⚡ Calculating optimal entry..."""
    
    await query.edit_message_text(processing_text, parse_mode="Markdown")
    
    try:
        # Get real binary recommendation from TwelveData
        recommendation = await twelvedata_client.get_binary_recommendation(symbol, expiry)
        
        # Format the binary options signal
        signal_text = format_binary_signal(recommendation)
        
        # Add action buttons
        keyboard = [
            [InlineKeyboardButton("🔄 New Signal", callback_data="menu_signals")],
            [InlineKeyboardButton("📈 Different Asset", callback_data="signal_assets")],
            [InlineKeyboardButton("💼 Account", callback_data="menu_account")]
        ]
        
        await query.edit_message_text(
            signal_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
        
        logger.info(f"✅ Binary signal sent for {symbol}")
        
    except Exception as e:
        logger.error(f"Binary signal error: {e}")
        await query.edit_message_text(
            "❌ **Signal Generation Failed**\n\n*Market data temporarily unavailable. Please try again.*",
            parse_mode="Markdown"
        )

def format_binary_signal(recommendation):
    """Format binary options signal professionally"""
    
    direction_emoji = "🟢" if recommendation['direction'] == 'CALL' else "🔴"
    confidence_emoji = "🎯" if recommendation['confidence'] > 70 else "⚡"
    
    return f"""
{direction_emoji} **BINARY OPTIONS SIGNAL** {direction_emoji}

📊 **Asset:** {recommendation['symbol']}
🎯 **Direction:** {recommendation['direction']}
⏰ **Expiry:** {recommendation['expiry']} minutes
💰 **Payout:** {recommendation['payout']}%
✅ **Confidence:** {recommendation['confidence']}% {confidence_emoji}

💵 **Current Price:** {recommendation['current_price']}
🕒 **Signal Time:** {recommendation['timestamp'][11:16]} UTC

🤖 **AI Analysis:**
{recommendation['analysis']}

📈 **Recommendation:**
• Binary Type: {recommendation['recommended_type']}
• Direction: **{recommendation['direction']}**
• Timeframe: {recommendation['expiry']} minutes
• Expected Payout: {recommendation['payout']}%

⚡ **Risk Management:**
• Only risk 1-2% per trade
• Use proper position sizing
• Trade during active sessions

🔍 *Data Source: {recommendation['data_source']}*
"""

async def handle_quick_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle quick binary signal request"""
    # Popular binary pairs with expiries
    keyboard = [
        [InlineKeyboardButton("EUR/USD - 5min", callback_data="signal_EURUSD_5")],
        [InlineKeyboardButton("GBP/USD - 5min", callback_data="signal_GBPUSD_5")],
        [InlineKeyboardButton("USD/JPY - 5min", callback_data="signal_USDJPY_5")],
        [InlineKeyboardButton("XAU/USD - 5min", callback_data="signal_XAUUSD_5")],
        [InlineKeyboardButton("BTC/USD - 5min", callback_data="signal_BTCUSD_5")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_main")]
    ]
    
    text = """
⚡ **Quick Binary Signals**

*Choose your asset and expiry:*

• **EUR/USD** - Most popular, high liquidity
• **GBP/USD** - Good volatility, clear trends  
• **USD/JPY** - Yen pairs, session dependent
• **XAU/USD** - Gold, safe haven asset
• **BTC/USD** - Crypto, high volatility

*All signals use real market data from TwelveData*"""
    
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
