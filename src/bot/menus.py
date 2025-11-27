import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from src.core.config import Config
from src.core.database import db

logger = logging.getLogger(__name__)

async def show_legal_disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show legal disclaimer before main menu"""
    disclaimer_text = """
⚠️ **LEGAL DISCLAIMER - IMPORTANT**

**RISK WARNING:**
Binary options trading involves substantial risk of loss and is not suitable for all investors. 
You should carefully consider whether trading is appropriate for you in light of your experience, 
objectives, financial resources and other relevant circumstances.

**IMPORTANT NOTICES:**
• This bot provides educational and informational signals only
• Past performance does not guarantee future results
• You may lose some or all of your invested capital
• Only trade with money you can afford to lose
• Seek independent financial advice if necessary

**BY CONTINUING YOU ACKNOWLEDGE:**
✓ You understand the risks involved
✓ You are aware that losses can exceed deposits
✓ You have read and accept this disclaimer
✓ You are at least 18 years old

*If you do not understand these risks, please do not continue.*"""

    keyboard = [
        [InlineKeyboardButton("✅ I UNDERSTAND & ACCEPT THE RISKS", callback_data="disclaimer_accepted")],
        [InlineKeyboardButton("❌ DECLINE & EXIT", callback_data="disclaimer_declined")]
    ]
    
    if update.callback_query:
        await update.callback_query.edit_message_text(
            disclaimer_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            disclaimer_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

async def show_binary_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show main menu optimized for binary options"""
    keyboard = [
        [InlineKeyboardButton("🎯 GET BINARY SIGNALS", callback_data="menu_signals")],
        [InlineKeyboardButton("📊 TRADING ASSETS", callback_data="menu_assets")],
        [InlineKeyboardButton("🤖 AI STRATEGIES", callback_data="menu_strategies")],
        [InlineKeyboardButton("💼 ACCOUNT & LIMITS", callback_data="menu_account")],
        [InlineKeyboardButton("📚 LEARN TRADING", callback_data="menu_education")],
        [InlineKeyboardButton("⚡ QUICK START GUIDE", callback_data="menu_quickstart")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = """
🤖 **Binary Options AI Pro** 🚀

*Professional AI-Powered Binary Trading*

🎯 **Live Binary Signals** with 85%+ Accuracy
📊 **15 Trading Assets** - Forex, Crypto, Commodities
🤖 **8 AI Engines** - Specialized for Binary Options
⏰ **Smart Expiry** - 1min to 60min Timeframes
💰 **Payout Calculator** - Based on Real Volatility

💎 **Your Account:** FREE TRIAL
📈 **Signals Today:** 0/3 Used
🕒 **Trial Ends:** 7 Days

*Tap buttons to start trading!*"""
    
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    else:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")

async def show_signals_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show binary signals menu with expiry options"""
    keyboard = [
        [InlineKeyboardButton("⚡ QUICK SIGNAL (5min EUR/USD)", callback_data="signal_EURUSD_5")],
        [InlineKeyboardButton("📈 STANDARD SIGNAL (15min EUR/USD)", callback_data="signal_EURUSD_15")],
        [InlineKeyboardButton("🎯 CUSTOM ASSET & EXPIRY", callback_data="menu_assets")],
        [InlineKeyboardButton("⚡ 1 MINUTE EXPIRY", callback_data="signal_EURUSD_1")],
        [InlineKeyboardButton("⚡ 2 MINUTES EXPIRY", callback_data="signal_EURUSD_2")],
        [InlineKeyboardButton("⚡ 5 MINUTES EXPIRY", callback_data="signal_EURUSD_5")],
        [InlineKeyboardButton("📈 15 MINUTES EXPIRY", callback_data="signal_EURUSD_15")],
        [InlineKeyboardButton("📈 30 MINUTES EXPIRY", callback_data="signal_EURUSD_30")],
        [InlineKeyboardButton("📈 60 MINUTES EXPIRY", callback_data="signal_EURUSD_60")],
        [InlineKeyboardButton("🔙 BACK TO MAIN MENU", callback_data="menu_main")]
    ]
    
    text = """
🎯 **Binary Options Signals**

*Choose your trading style:*

⚡ **Quick Signal** - 5min expiry, fast action
📈 **Standard** - 15min expiry, more analysis time
🎯 **Custom** - Choose asset & expiry

**Popular Expiry Times:**
• 1-2min - Scalping, high frequency
• 5-15min - Intraday, balanced
• 30-60min - Swing, more analysis

*All signals use real TwelveData market feeds*"""
    
    await update.callback_query.edit_message_text(
        text, 
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )

async def show_asset_expiry_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, asset: str):
    """Show expiry options for a specific asset"""
    keyboard = [
        [InlineKeyboardButton("⚡ 1 MINUTE EXPIRY", callback_data=f"expiry_{asset}_1")],
        [InlineKeyboardButton("⚡ 2 MINUTES EXPIRY", callback_data=f"expiry_{asset}_2")],
        [InlineKeyboardButton("⚡ 5 MINUTES EXPIRY", callback_data=f"expiry_{asset}_5")],
        [InlineKeyboardButton("📈 15 MINUTES EXPIRY", callback_data=f"expiry_{asset}_15")],
        [InlineKeyboardButton("📈 30 MINUTES EXPIRY", callback_data=f"expiry_{asset}_30")],
        [InlineKeyboardButton("📈 60 MINUTES EXPIRY", callback_data=f"expiry_{asset}_60")],
        [InlineKeyboardButton("🔙 BACK TO ASSETS", callback_data="menu_assets")],
        [InlineKeyboardButton("🔙 BACK TO MAIN MENU", callback_data="menu_main")]
    ]
    
    # Get asset info
    asset_type = "💱 Forex" if "/" in asset and "XAU" not in asset else "₿ Crypto" if "BTC" in asset or "ETH" in asset else "🟡 Commodity" if "XAU" in asset else "📈 Index"
    
    text = f"""
{asset_type} **{asset}** - Binary Options

*Choose expiry time for your trade:*

⚡ **1-5 minutes** - Quick trades, fast results
📊 **15-30 minutes** - More analysis time  
📈 **60 minutes** - Swing trading approach

**Recommended for {asset}:**
• High volatility: Shorter expiries (1-5min)
• Trending markets: Medium expiries (15-30min)  
• Range markets: Longer expiries (30-60min)

*AI will analyze current market conditions*"""
    
    await update.callback_query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )

async def show_strategies_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show binary trading strategies"""
    keyboard = [
        [InlineKeyboardButton("🚀 TREND FOLLOWING STRATEGY", callback_data="strategy_trend")],
        [InlineKeyboardButton("⚡ MEAN REVERSION STRATEGY", callback_data="strategy_meanreversion")],
        [InlineKeyboardButton("📊 BREAKOUT TRADING STRATEGY", callback_data="strategy_breakout")],
        [InlineKeyboardButton("🎯 VOLATILITY ANALYSIS STRATEGY", callback_data="strategy_volatility")],
        [InlineKeyboardButton("🔙 BACK TO MAIN MENU", callback_data="menu_main")]
    ]
    
    text = """
🤖 **AI Trading Strategies**

*Choose your binary options strategy:*

🚀 **Trend Following**
- Trade with the current trend
- Use moving averages & momentum
- Best during strong trends

⚡ **Mean Reversion**  
- Trade against extremes
- Use RSI & Bollinger Bands
- Best in ranging markets

📊 **Breakout Trading**
- Trade breakouts from ranges
- Use support/resistance levels
- Best during volatility bursts

🎯 **Volatility Analysis**
- Trade based on volatility
- Use ATR & volatility indicators
- Best during news events

*Each strategy uses different AI engines*"""
    
    await update.callback_query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )

async def show_strategy_info(update: Update, context: ContextTypes.DEFAULT_TYPE, strategy: str):
    """Show detailed strategy information"""
    strategy_info = {
        "strategy_trend": """
🚀 **Trend Following Strategy**

*AI analyzes market trends and momentum*

**How it works:**
1. Identifies current trend direction
2. Uses multiple timeframes for confirmation  
3. Enters trades in trend direction
4. Uses trailing logic for exits

**Best for:**
- Strong trending markets
- Major currency pairs
- London/NY session overlaps

**AI Engines Used:**
- Trend Analysis AI
- Momentum Detection
- Multi-timeframe Analysis

**Recommended Assets:**
- EUR/USD, GBP/USD, USD/JPY
- During London (7:00-16:00 UTC) and New York (12:00-21:00 UTC) sessions""",

        "strategy_meanreversion": """
⚡ **Mean Reversion Strategy**

*AI identifies overbought/oversold conditions*

**How it works:**
1. Detects price extremes using RSI
2. Uses Bollinger Bands for levels
3. Enters when price reverts to mean
4. Quick 1-5 minute trades

**Best for:**
- Ranging markets
- Asian session trading
- Low volatility periods

**AI Engines Used:**
- RSI Analysis AI
- Statistical Mean Reversion
- Volatility Assessment

**Recommended Assets:**
- EUR/GBP, USD/CHF, XAU/USD
- During Asian session (22:00-6:00 UTC)""",

        "strategy_breakout": """
📊 **Breakout Trading Strategy**

*AI detects breakouts from consolidation*

**How it works:**
1. Identifies consolidation patterns
2. Monitors key support/resistance
3. Enters on confirmed breakouts
4. Uses volume confirmation

**Best for:**
- News events
- Session openings
- High volatility periods

**AI Engines Used:**
- Pattern Recognition AI
- Volume Analysis
- Breakout Detection

**Recommended Assets:**
- GBP/JPY, BTC/USD, US30
- During news releases and session overlaps""",

        "strategy_volatility": """
🎯 **Volatility Analysis Strategy**

*AI trades based on volatility conditions*

**How it works:**
1. Measures current volatility levels
2. Compares to historical averages
3. Adjusts expiry times based on volatility
4. Uses ATR and volatility indicators

**Best for:**
- Economic news releases
- Market openings/closings
- Unexpected volatility spikes

**AI Engines Used:**
- Volatility AI
- ATR Analysis
- News Impact Assessment

**Recommended Assets:**
- All assets during high volatility
- News trading with proper risk management"""
    }
    
    info = strategy_info.get(strategy, "Strategy information coming soon.")
    
    keyboard = [
        [InlineKeyboardButton("🎯 USE THIS STRATEGY NOW", callback_data="menu_signals")],
        [InlineKeyboardButton("📊 VIEW ALL STRATEGIES", callback_data="menu_strategies")],
        [InlineKeyboardButton("🔙 BACK TO MAIN MENU", callback_data="menu_main")]
    ]
    
    await update.callback_query.edit_message_text(
        info,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )

async def show_account_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show account management menu"""
    user = update.effective_user
    user_data = db.get_user(user.id) if hasattr(db, 'get_user') else None
    
    keyboard = [
        [InlineKeyboardButton("💎 UPGRADE PLAN", callback_data="account_upgrade")],
        [InlineKeyboardButton("📊 USAGE STATISTICS", callback_data="account_stats")],
        [InlineKeyboardButton("🆓 TRIAL INFORMATION", callback_data="account_trial")],
        [InlineKeyboardButton("🔧 SETTINGS", callback_data="account_settings")],
        [InlineKeyboardButton("🔙 BACK TO MAIN MENU", callback_data="menu_main")]
    ]
    
    text = f"""
💼 **Binary Trading Account**

👤 **Trader:** {user.first_name}
🆔 **ID:** {user.id}
📅 **Member Since:** Recent

**Subscription:** 🆓 FREE TRIAL
**Signals Used:** 0/3 Today
**AI Access:** 8 Engines Active
**Assets Available:** 15 Pairs

**Trial Benefits:**
✓ 3 Signals Per Day
✓ All 15 Trading Assets  
✓ 8 AI Engines
✓ Real Market Data
✓ Basic Strategies

💎 **Upgrade for:**
• Unlimited Signals
• Advanced AI Engines
• Premium Strategies
• Priority Support
• Advanced Analytics

*Contact admin to upgrade your plan*"""
    
    await update.callback_query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )

async def show_education_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show binary options education menu"""
    keyboard = [
        [InlineKeyboardButton("📚 BINARY BASICS", callback_data="edu_basics")],
        [InlineKeyboardButton("🎯 RISK MANAGEMENT", callback_data="edu_risk")],
        [InlineKeyboardButton("🤖 HOW TO USE THIS BOT", callback_data="edu_bot_usage")],
        [InlineKeyboardButton("📊 TECHNICAL ANALYSIS", callback_data="edu_technical")],
        [InlineKeyboardButton("💡 TRADING PSYCHOLOGY", callback_data="edu_psychology")],
        [InlineKeyboardButton("🔙 BACK TO MAIN MENU", callback_data="menu_main")]
    ]
    
    text = """
📚 **Binary Options Education**

*Learn professional binary trading:*

📚 **Binary Basics** - How binary options work
🎯 **Risk Management** - Protect your capital
🤖 **How to Use This Bot** - Step-by-step guide
📊 **Technical Analysis** - Reading charts & indicators
💡 **Trading Psychology** - Master your mindset

**Essential Knowledge:**
• Understanding CALL/PUT options
• Expiry times and their impact
• Volatility and payout relationships
• Risk management principles

*Knowledge is power in trading*"""
    
    await update.callback_query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )

async def show_quickstart_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show quick start guide"""
    keyboard = [
        [InlineKeyboardButton("🎯 GET YOUR FIRST SIGNAL", callback_data="signal_EURUSD_5")],
        [InlineKeyboardButton("📊 VIEW ALL ASSETS", callback_data="menu_assets")],
        [InlineKeyboardButton("🤖 LEARN STRATEGIES", callback_data="menu_strategies")],
        [InlineKeyboardButton("🔙 BACK TO MAIN MENU", callback_data="menu_main")]
    ]
    
    text = """
🚀 **Quick Start Guide**

*Follow these 5 simple steps:*

1. **🎯 GET SIGNALS** - Click above to get your first signal
2. **📊 CHOOSE ASSET** - Select from 15 trading instruments
3. **⏰ PICK EXPIRY** - 1min to 60min timeframes
4. **🤖 AI ANALYSIS** - Get CALL/PUT prediction with confidence
5. **💰 PLACE TRADE** - Execute on your preferred platform

**For Beginners:**
• Start with EUR/USD 5min signals
• Use demo account first
• Risk only 1-2% per trade
• Trade during London/NY sessions (7:00-16:00 UTC)

*Ready to start? Click the button below!*"""
    
    await update.callback_query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )
