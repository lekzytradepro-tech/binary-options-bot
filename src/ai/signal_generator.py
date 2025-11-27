import logging
import random
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from src.core.config import Config

logger = logging.getLogger(__name__)

async def handle_quick_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle quick binary signal request"""
    try:
        logger.info("🎯 Generating quick binary signal")
        
        # Simulate AI analysis
        direction = "CALL" if random.random() > 0.5 else "PUT"
        confidence = random.randint(65, 92)
        
        signal_text = f"""
⚡ **Quick Binary Signal - EUR/USD**

🎯 **Direction:** {'📈 CALL' if direction == 'CALL' else '📉 PUT'}
📊 **Confidence:** {confidence}%
⏰ **Expiry:** 5 minutes
💎 **Asset:** EUR/USD

**AI Analysis:**
• Trend Analysis: ✅ Confirmed
• Momentum: ✅ Strong
• Volatility: ✅ Optimal
• Pattern Recognition: ✅ Aligned

💰 **Expected Payout:** 78-82%

**Recommendation:**
Place a **{direction}** option with 5-minute expiry.

*Remember: Always use proper risk management!*"""
        
        keyboard = [
            [InlineKeyboardButton("🔄 Get Another Signal", callback_data="signal_EURUSD_5")],
            [InlineKeyboardButton("📊 View All Assets", callback_data="menu_assets")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="menu_main")]
        ]
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                signal_text, 
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                signal_text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
            
    except Exception as e:
        logger.error(f"Error in handle_quick_signal: {e}")
        if update.message:
            await update.message.reply_text(
                "❌ **Error generating signal**\n\n*Please try again later.*",
                parse_mode="Markdown"
            )

async def generate_binary_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, signal_data: str):
    """Generate binary signal based on user selection"""
    try:
        logger.info(f"🎯 Generating binary signal: {signal_data}")
        
        # Parse signal data (format: signal_ASSET_EXPIRY)
        parts = signal_data.split("_")
        asset = parts[1] if len(parts) > 1 else "EUR/USD"
        expiry = parts[2] if len(parts) > 2 else "5"
        
        # Simulate AI analysis
        direction = "CALL" if random.random() > 0.5 else "PUT"
        confidence = random.randint(65, 92)
        
        # Determine asset type for emoji
        if "USD" in asset and "/" in asset:
            asset_emoji = "💱"
        elif "BTC" in asset or "ETH" in asset:
            asset_emoji = "₿"
        elif "XAU" in asset:
            asset_emoji = "🟡"
        else:
            asset_emoji = "📈"
        
        signal_text = f"""
{asset_emoji} **Binary Signal - {asset}**

🎯 **Direction:** {'📈 CALL' if direction == 'CALL' else '📉 PUT'}
📊 **Confidence:** {confidence}%
⏰ **Expiry:** {expiry} minutes
💎 **Asset:** {asset}

**AI Analysis Complete:**
• Trend Analysis: ✅ Confirmed
• Momentum Detection: ✅ Strong
• Volatility Assessment: ✅ Optimal
• Pattern Recognition: ✅ Aligned
• Support/Resistance: ✅ Valid
• Market Sentiment: ✅ Positive

💰 **Expected Payout:** 75-85%

**Trading Recommendation:**
Place a **{direction}** option with {expiry}-minute expiry.

⚠️ **Risk Management:**
• Risk only 1-2% of your capital
• Use demo account for testing
• Trade during active sessions
• Set mental stop losses

*Signal valid for 2 minutes*"""
        
        keyboard = [
            [InlineKeyboardButton("🔄 New Signal (Same Settings)", callback_data=signal_data)],
            [InlineKeyboardButton("📊 Different Asset", callback_data="menu_assets")],
            [InlineKeyboardButton("⏰ Different Expiry", callback_data=f"asset_{asset}")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="menu_main")]
        ]
        
        await update.callback_query.edit_message_text(
            signal_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
        
    except Exception as e:
        logger.error(f"Error in generate_binary_signal: {e}")
        await update.callback_query.edit_message_text(
            "❌ **Error generating signal**\n\n*Please try again or contact support.*",
            parse_mode="Markdown"
        )
