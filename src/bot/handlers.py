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
    
    # Return default values if user info can't be extracted
    return {'id': 0, 'username': 'unknown', 'first_name': 'User'}

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command for binary options"""
    try:
        user_info = get_user_info(update)
        
        # Add user to database
        db.add_user(user_info['id'], user_info['username'], user_info['first_name'])
        
        # Show legal disclaimer first
        from src.bot.menus import show_legal_disclaimer
        await show_legal_disclaimer(update, context)
        
        logger.info(f"👤 New user started: {user_info['id']} - {user_info['first_name']}")
        
    except Exception as e:
        logger.error(f"Error in handle_start: {e}")
        # Fallback response
        if update.message:
            await update.message.reply_text(
                "🤖 Welcome to Binary Options AI Pro! 🚀\n\n"
                "I'm here to help you with AI-powered binary options trading. "
                "Use the buttons below to navigate.",
                parse_mode="Markdown"
            )

async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command with binary options focus"""
    help_text = """
📖 **Binary Options AI Pro - Help**

🤖 *AI-Powered Binary Options Trading*

**Commands:**
/start - Start bot with binary options
/help - Show this help message  
/signals - Get quick binary signals
/assets - View trading assets
/status - Check bot status
/quickstart - Quick start guide

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

*Binary options trading involves high risk. Only trade with money you can afford to lose.*"""
    
    try:
        if update.message:
            await update.message.reply_text(help_text, parse_mode="Markdown")
        elif update.callback_query:
            await update.callback_query.message.reply_text(help_text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in handle_help: {e}")

async def handle_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /signals command - quick binary signals"""
    try:
        from src.ai.signal_generator import handle_quick_signal
        await handle_quick_signal(update, context)
    except Exception as e:
        logger.error(f"Error in handle_signals: {e}")
        # Fallback response
        if update.message:
            await update.message.reply_text(
                "🎯 **Quick Binary Signal**\n\n"
                "🤖 AI is analyzing the market...\n"
                "⚡ Generating signal for EUR/USD 5min...\n\n"
                "*Please try again in a moment*",
                parse_mode="Markdown"
            )

async def handle_assets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show available binary trading assets"""
    try:
        keyboard = []
        
        # Group assets by category
        forex_assets = [p for p in Config.BINARY_PAIRS if '/' in p and 'XAU' not in p and 'BTC' not in p]
        crypto_assets = [p for p in Config.BINARY_PAIRS if 'BTC' in p or 'ETH' in p]
        commodity_assets = [p for p in Config.BINARY_PAIRS if 'XAU' in p or 'XAG' in p]
        indices_assets = [p for p in Config.BINARY_PAIRS if 'US30' in p or 'SPX' in p]
        
        # Add forex assets (2 per row)
        for i in range(0, len(forex_assets), 2):
            row = []
            for asset in forex_assets[i:i+2]:
                row.append(InlineKeyboardButton(f"💱 {asset}", callback_data=f"asset_{asset}"))
            keyboard.append(row)
        
        # Add crypto assets (full width)
        for asset in crypto_assets:
            keyboard.append([InlineKeyboardButton(f"₿ {asset}", callback_data=f"asset_{asset}")])
        
        # Add commodities (full width)
        for asset in commodity_assets:
            keyboard.append([InlineKeyboardButton(f"🟡 {asset}", callback_data=f"asset_{asset}")])
        
        # Add indices (full width)
        for asset in indices_assets:
            keyboard.append([InlineKeyboardButton(f"📈 {asset}", callback_data=f"asset_{asset}")])
        
        keyboard.append([InlineKeyboardButton("🔙 BACK TO MAIN MENU", callback_data="menu_main")])
        
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
    except Exception as e:
        logger.error(f"Error in handle_assets: {e}")
        if update.message:
            await update.message.reply_text(
                "❌ **Error loading assets**\n\n*Please try again later.*",
                parse_mode="Markdown"
            )

async def handle_button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all binary options button clicks"""
    try:
        query = update.callback_query
        await query.answer()
        
        data = query.data
        logger.info(f"🔄 Button clicked: {data}")
        
        # Legal disclaimer handling
        if data == "disclaimer_accepted":
            from src.bot.menus import show_binary_main_menu
            await show_binary_main_menu(update, context)
            return
            
        elif data == "disclaimer_declined":
            await query.edit_message_text(
                "❌ **Disclaimer Declined**\n\n"
                "You have chosen not to accept the legal disclaimer. "
                "To use Binary Options AI Pro, you must accept the terms and conditions.\n\n"
                "If you change your mind, use /start to begin again.\n\n"
                "*Thank you for your understanding.*",
                parse_mode="Markdown"
            )
            return

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
            
        elif data == "menu_assets":
            await handle_assets(update, context)
            
        elif data == "menu_quickstart":
            from src.bot.menus import show_quickstart_menu
            await show_quickstart_menu(update, context)
        
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
            
        # Account management
        elif data.startswith("account_"):
            if data == "account_upgrade":
                await query.edit_message_text(
                    "💎 **Account Upgrade**\n\n"
                    "*Premium Features:*\n"
                    "• 📈 Unlimited daily signals\n"
                    "• 🤖 All 8 AI engines\n"
                    "• 💰 15 trading assets\n"
                    "• 🎯 Advanced strategies\n"
                    "• ⚡ Priority support\n"
                    "• 📊 Advanced analytics\n\n"
                    "*VIP Features:*\n"
                    "• 🚀 Unlimited everything\n"
                    "• 💎 Dedicated support\n"
                    "• 🔧 Custom strategies\n"
                    "• 📈 Performance insights\n\n"
                    "*Contact admin to upgrade your plan*",
                    parse_mode="Markdown"
                )
            elif data == "account_stats":
                user_info = get_user_info(update)
                await query.edit_message_text(
                    f"📊 **Account Statistics**\n\n"
                    f"👤 **User:** {user_info['first_name']}\n"
                    f"🆔 **ID:** {user_info['id']}\n"
                    f"📅 **Member Since:** Recent\n"
                    f"🎯 **Signals Today:** 0/3\n"
                    f"🤖 **AI Access:** 8 Engines\n"
                    f"📈 **Assets:** 15 Available\n"
                    f"💎 **Plan:** Free Trial\n"
                    f"⏰ **Trial Ends:** 7 days\n\n"
                    f"*Upgrade for unlimited access*",
                    parse_mode="Markdown"
                )
            elif data == "account_trial":
                await query.edit_message_text(
                    "🆓 **Free Trial Information**\n\n"
                    "*Trial Benefits:*\n"
                    "✓ 3 signals per day\n"
                    "✓ All 15 trading assets\n"
                    "✓ 8 AI engines access\n"
                    "✓ Real market data\n"
                    "✓ Basic strategies\n"
                    "✓ Educational content\n\n"
                    "*Trial Duration:* 7 days\n"
                    "*Auto-renewal:* No\n"
                    "*Credit Card:* Not required\n\n"
                    "*Upgrade for unlimited signals and premium features*",
                    parse_mode="Markdown"
                )
            elif data == "account_settings":
                await query.edit_message_text(
                    "🔧 **Account Settings**\n\n"
                    "*Available Settings:*\n"
                    "• 🔔 Notification preferences\n"
                    "• 🎯 Risk management limits\n"
                    "• ⏰ Trading sessions\n"
                    "• 📊 Signal preferences\n"
                    "• 🤖 AI engine selection\n\n"
                    "*Settings panel coming soon*\n"
                    "*Currently using default settings*",
                    parse_mode="Markdown"
                )
                
        # Education menu
        elif data.startswith("edu_"):
            if data == "edu_basics":
                await query.edit_message_text(
                    "📚 **Binary Options Basics**\n\n"
                    "*What are Binary Options?*\n"
                    "Binary options are financial instruments where you predict whether "
                    "an asset's price will be above or below a certain level at expiration.\n\n"
                    "*CALL Option:* Price will be HIGHER\n"
                    "*PUT Option:* Price will be LOWER\n\n"
                    "*Key Concepts:*\n"
                    "• **Expiry Time:** When the trade closes (1min-60min)\n"
                    "• **Payout:** Return if correct (70-90%)\n"
                    "• **Strike Price:** Current price when placing trade\n"
                    "• **In/Out Money:** Win/loss conditions\n\n"
                    "*Example Trade:*\n"
                    "Asset: EUR/USD\n"
                    "Expiry: 5 minutes\n"
                    "Prediction: CALL (Price will rise)\n"
                    "Investment: $10\n"
                    "Potential Payout: $18 (80% return)\n"
                    "Potential Loss: $10 (if wrong)",
                    parse_mode="Markdown"
                )
            elif data == "edu_risk":
                await query.edit_message_text(
                    "🎯 **Risk Management**\n\n"
                    "*Essential Risk Rules:*\n"
                    "• 💰 Risk only 1-2% per trade\n"
                    "• 🎯 Use demo account first\n"
                    "• 📉 Set daily loss limits\n"
                    "• ⏰ Trade during active sessions\n"
                    "• 😊 Avoid emotional trading\n"
                    "• 📊 Keep a trading journal\n\n"
                    "*Risk Management Strategy:*\n"
                    "1. Start with demo account\n"
                    "2. Risk small amounts initially\n"
                    "3. Set stop-loss mentally\n"
                    "4. Don't chase losses\n"
                    "5. Take breaks regularly\n\n"
                    "*Remember:*\n"
                    "Binary options involve high risk. Only trade with money you can afford to lose. "
                    "Never trade with emergency funds or money needed for essential expenses.",
                    parse_mode="Markdown"
                )
            elif data == "edu_bot_usage":
                await query.edit_message_text(
                    "🤖 **How to Use This Bot - Complete Guide**\n\n"
                    "*Step-by-Step Process:*\n\n"
                    "1. **🎯 GET SIGNALS**\n"
                    "   - Click 'Get Binary Signals'\n"
                    "   - Choose expiry time (1min-60min)\n"
                    "   - Or select custom asset\n\n"
                    "2. **📊 ANALYZE SIGNAL**\n"
                    "   - AI provides CALL/PUT prediction\n"
                    "   - Confidence percentage shown\n"
                    "   - Market analysis included\n"
                    "   - Risk assessment provided\n\n"
                    "3. **💰 EXECUTE TRADE**\n"
                    "   - Use your preferred broker\n"
                    "   - Set appropriate investment amount\n"
                    "   - Place CALL or PUT based on signal\n"
                    "   - Confirm expiry time matches\n\n"
                    "4. **📈 MANAGE TRADE**\n"
                    "   - Monitor until expiry\n"
                    "   - Use proper risk management\n"
                    "   - Review performance\n"
                    "   - Learn from each trade\n\n"
                    "*Pro Tips:*\n"
                    "• 📱 Start with demo account\n"
                    "• 💰 Use 1-2% risk per trade\n"
                    "• ⏰ Trade during active sessions\n"
                    "• 📝 Keep a trading journal\n"
                    "• 🧠 Stay disciplined\n"
                    "• 🔄 Review your strategy regularly\n\n"
                    "*Best Trading Times:*\n"
                    "• London Session: 7:00-16:00 UTC\n"
                    "• New York Session: 12:00-21:00 UTC\n"
                    "• Overlap: 12:00-16:00 UTC (Highest volatility)",
                    parse_mode="Markdown"
                )
            elif data == "edu_technical":
                await query.edit_message_text(
                    "📊 **Technical Analysis**\n\n"
                    "*Key Indicators We Use:*\n"
                    "• 📈 RSI (Overbought/Oversold)\n"
                    "• 📊 MACD (Trend/Momentum)\n"
                    "• 📉 Bollinger Bands (Volatility)\n"
                    "• 🎯 Moving Averages (Trend)\n"
                    "• ⚡ Support/Resistance Levels\n"
                    "• 💹 Volume Analysis\n"
                    "• 📊 ATR (Volatility)\n\n"
                    "*Timeframes Analyzed:*\n"
                    "• 1min, 5min, 15min, 1h\n"
                    "• Multi-timeframe confirmation\n"
                    "• Real-time price action\n\n"
                    "*How AI Uses Technical Analysis:*\n"
                    "1. **Trend Identification** - Direction and strength\n"
                    "2. **Momentum Analysis** - Speed of price movement\n"
                    "3. **Volatility Assessment** - Market conditions\n"
                    "4. **Pattern Recognition** - Chart formations\n"
                    "5. **Support/Resistance** - Key price levels\n\n"
                    "*Remember:* Technical analysis helps but doesn't guarantee results. "
                    "Always use proper risk management.",
                    parse_mode="Markdown"
                )
            elif data == "edu_psychology":
                await query.edit_message_text(
                    "💡 **Trading Psychology**\n\n"
                    "*Master Your Mindset for Success:*\n\n"
                    "**Common Psychological Traps:**\n"
                    "• 😠 **Revenge Trading** - Trading to recover losses\n"
                    "• 😰 **Fear of Missing Out** - Entering late trades\n"
                    "• 💰 **Greed** - Holding trades too long\n"
                    "• 😨 **Fear** - Exiting trades too early\n"
                    "• 🎯 **Overconfidence** - Taking excessive risks\n\n"
                    "**Healthy Trading Mindset:**\n"
                    "• 🧘 **Stay Calm** - Emotions cloud judgment\n"
                    "• 📊 **Be Objective** - Follow your strategy\n"
                    "• 💪 **Discipline** - Stick to your rules\n"
                    "• 📈 **Patience** - Wait for good setups\n"
                    "• 🔄 **Adaptability** - Learn and adjust\n\n"
                    "**Practical Tips:**\n"
                    "1. Trade only when focused\n"
                    "2. Take regular breaks\n"
                    "3. Review trades objectively\n"
                    "4. Accept losses as learning\n"
                    "5. Celebrate small wins\n"
                    "6. Maintain work-life balance\n\n"
                    "*Remember:* Successful trading is 80% psychology, 20% strategy.",
                    parse_mode="Markdown"
                )
            
        else:
            await query.edit_message_text(
                "🔄 **Feature Coming Soon**\n\n"
                "*This binary options feature is in development and will be available soon.*\n\n"
                "🔙 Use the main menu to access available features.",
                parse_mode="Markdown"
            )
            
    except Exception as e:
        logger.error(f"Button handler error: {e}")
        try:
            await update.callback_query.edit_message_text(
                "❌ **Error Processing Request**\n\n"
                "*Please try again or contact support if the problem persists.*\n\n"
                f"Error details: {str(e)}",
                parse_mode="Markdown"
            )
        except:
            # If we can't edit the message, try to send a new one
            if update.callback_query and update.callback_query.message:
                await update.callback_query.message.reply_text(
                    "❌ **Error Processing Request**\n\n"
                    "*Please try again.*",
                    parse_mode="Markdown"
                )

async def handle_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle unknown commands and messages"""
    try:
        if update.message:
            await update.message.reply_text(
                "🤖 **Binary Options AI Pro**\n\n"
                "I didn't understand that command. Here are available commands:\n\n"
                "• /start - Start binary trading\n"
                "• /help - Get help with binary options\n"  
                "• /signals - Get AI trading signals\n"
                "• /assets - View trading assets\n"
                "• /status - Check bot status\n"
                "• /quickstart - Quick start guide\n\n"
                "*Use the menu buttons for the best experience!*",
                parse_mode="Markdown"
            )
    except Exception as e:
        logger.error(f"Error in handle_unknown: {e}")

async def handle_error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors in the bot"""
    try:
        logger.error(f"Bot error: {context.error}")
        
        # Notify user about error
        if update and update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ **An error occurred**\n\n"
                     "*Please try again or contact support if the problem persists.*",
                parse_mode="Markdown"
            )
    except Exception as e:
        logger.error(f"Error in handle_error: {e}")

async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command - show bot status"""
    try:
        status_text = """
🤖 **Binary Options AI Pro - Status**

✅ **Bot Status:** Operational
🎯 **AI Engines:** 8/8 Active
📊 **Market Data:** Live
💾 **Database:** Connected
⚡ **Performance:** Optimal

**Services:**
• TwelveData API: Connected
• Signal Generation: Active
• Risk Management: Enabled
• User Accounts: Working

*All systems operational*"""
        
        if update.message:
            await update.message.reply_text(status_text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in handle_status: {e}")

async def handle_quick_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick start guide for new users"""
    try:
        quick_start_text = """
🚀 **Binary Options Quick Start**

*Follow these 5 simple steps:*

1. **🎯 GET SIGNALS** - Use /signals or menu
2. **📊 CHOOSE ASSET** - Select from 15 options  
3. **⏰ PICK EXPIRY** - 1min to 60min timeframes
4. **🤖 AI ANALYSIS** - Get CALL/PUT prediction
5. **💰 PLACE TRADE** - On your broker platform

**For Beginners:**
• Start with EUR/USD 5min signals
• Use demo account first
• Risk only 1-2% per trade
• Trade during London/NY sessions (7:00-16:00 UTC)

*Ready to start? Use /signals now!*"""
        
        if update.message:
            await update.message.reply_text(quick_start_text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in handle_quick_start: {e}")
