import logging
import os
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebhookBot:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.webhook_url = os.getenv("WEBHOOK_URL")
        self.application = None
        
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        if not self.webhook_url:
            raise ValueError("WEBHOOK_URL is required")

    async def initialize(self):
        """Initialize the bot application"""
        self.application = (
            Application.builder()
            .token(self.token)
            .build()
        )
        
        self.setup_handlers()
        await self.set_webhook()
        
        logger.info("🤖 Bot initialized successfully")
        logger.info(f"🌐 Webhook URL: {self.webhook_url}")

    def setup_handlers(self):
        """Setup all bot handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.handle_start))
        self.application.add_handler(CommandHandler("help", self.handle_help))
        self.application.add_handler(CommandHandler("signals", self.handle_signals))
        
        # Button handlers
        self.application.add_handler(CallbackQueryHandler(self.handle_buttons))
        
        logger.info("✅ Bot handlers setup completed")

    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        
        keyboard = [
            [InlineKeyboardButton("🚀 Get AI Signals", callback_data="menu_signals")],
            [InlineKeyboardButton("📊 Trading Strategies", callback_data="menu_strategies")],
            [InlineKeyboardButton("💼 Account Dashboard", callback_data="menu_account")],
            [InlineKeyboardButton("📚 Learn Trading", callback_data="menu_education")],
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_text = f"""
🤖 **Binary Options AI Pro** 🚀

Welcome *{user.first_name}*! 

I provide **AI-powered trading signals** with professional analysis.

🎯 **Core Features:**
• 15 AI Engines for signal generation
• Real-time market analysis  
• Multiple trading strategies
• Professional risk management
• Educational trading content

💎 **Account:** Free Trial (3 signals/day)
📊 **Status:** ✅ Active

*Use the buttons below to navigate*"""
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
        logger.info(f"User {user.id} started bot")

    async def handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
📖 **Binary Options AI Pro - Help**

**Available Commands:**
/start - Start the bot and show main menu
/help - Show this help message  
/signals - Get quick trading signals

**Navigation:**
• Use buttons for all features
• No commands needed for most actions

**Support:**
Contact admin for assistance"""
        
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def handle_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command"""
        keyboard = [
            [InlineKeyboardButton("⚡ Quick Signal (1-5min)", callback_data="signal_quick")],
            [InlineKeyboardButton("📈 Trend Analysis", callback_data="signal_trend")],
            [InlineKeyboardButton("🎯 Pattern Scanner", callback_data="signal_pattern")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="menu_main")]
        ]
        
        await update.message.reply_text(
            "📊 **Quick Signal Access**\n\nChoose your signal type:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    async def handle_buttons(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle all button clicks"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        # Main menu navigation
        if data == "menu_signals":
            await self.show_signals_menu(query)
        elif data == "menu_strategies":
            await self.show_strategies_menu(query)
        elif data == "menu_account":
            await self.show_account_menu(query)
        elif data == "menu_education":
            await self.show_education_menu(query)
        elif data == "menu_main":
            await self.show_main_menu(query)
        
        # Signal types
        elif data.startswith("signal_"):
            await self.handle_signal_request(query, data)
            
        # Strategy types  
        elif data.startswith("strategy_"):
            await self.handle_strategy_request(query, data)
            
        else:
            await query.edit_message_text("🔄 Feature coming soon!")

    async def show_main_menu(self, query):
        """Show main menu"""
        keyboard = [
            [InlineKeyboardButton("🚀 Get AI Signals", callback_data="menu_signals")],
            [InlineKeyboardButton("📊 Trading Strategies", callback_data="menu_strategies")],
            [InlineKeyboardButton("💼 Account Dashboard", callback_data="menu_account")],
            [InlineKeyboardButton("📚 Learn Trading", callback_data="menu_education")],
        ]
        
        await query.edit_message_text(
            "🤖 **Binary Options AI Pro** - Main Menu\n\n*Choose your action:*",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    async def show_signals_menu(self, query):
        """Show signals menu"""
        keyboard = [
            [InlineKeyboardButton("⚡ Quick Signal (1-5min)", callback_data="signal_quick")],
            [InlineKeyboardButton("📈 Trend Analysis (5-15min)", callback_data="signal_trend")],
            [InlineKeyboardButton("🎯 Pattern Scanner", callback_data="signal_pattern")],
            [InlineKeyboardButton("📊 Volume Analysis", callback_data="signal_volume")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="menu_main")]
        ]
        
        await query.edit_message_text(
            "📊 **AI Trading Signals**\n\n*Choose your signal type:*\n\n⚡ **Quick** - Fast 1-5 minute trades\n📈 **Trend** - Directional trend analysis\n🎯 **Pattern** - AI pattern recognition\n📊 **Volume** - Volume-based signals",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    async def show_strategies_menu(self, query):
        """Show strategies menu"""
        keyboard = [
            [InlineKeyboardButton("🚀 Trend Spotter", callback_data="strategy_trend")],
            [InlineKeyboardButton("⚡ Scalper Pro", callback_data="strategy_scalper")],
            [InlineKeyboardButton("📊 Volume Analysis", callback_data="strategy_volume")],
            [InlineKeyboardButton("🎯 Pattern Master", callback_data="strategy_pattern")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="menu_main")]
        ]
        
        await query.edit_message_text(
            "🎯 **Trading Strategies**\n\n*Choose your trading style:*\n\n🚀 **Trend Spotter** - Follow market trends\n⚡ **Scalper Pro** - Quick 1-3 minute trades\n📊 **Volume Analysis** - Volume-based entries\n🎯 **Pattern Master** - Pattern recognition",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    async def show_account_menu(self, query):
        """Show account menu"""
        user = query.from_user
        
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade Plan", callback_data="account_upgrade")],
            [InlineKeyboardButton("📊 Usage Statistics", callback_data="account_stats")],
            [InlineKeyboardButton("🆓 Free Trial Info", callback_data="account_trial")],
            [InlineKeyboardButton("🔧 Settings", callback_data="account_settings")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="menu_main")]
        ]
        
        account_text = f"""
💼 **Account Management**

👤 **User:** {user.first_name}
🆔 **ID:** {user.id}
📅 **Status:** ✅ Active

**Subscription:** 🆓 Free Trial
**Signals Today:** 0/3 used
**Expiry:** 7 days remaining

💎 **Upgrade for:**
• Unlimited signals
• All AI strategies  
• Priority processing
• Advanced features"""

        await query.edit_message_text(
            account_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    async def show_education_menu(self, query):
        """Show education menu"""
        keyboard = [
            [InlineKeyboardButton("📚 Trading Basics", callback_data="edu_basics")],
            [InlineKeyboardButton("🎯 Risk Management", callback_data="edu_risk")],
            [InlineKeyboardButton("🤖 AI Strategies", callback_data="edu_ai")],
            [InlineKeyboardButton("📊 Technical Analysis", callback_data="edu_technical")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="menu_main")]
        ]
        
        await query.edit_message_text(
            "📚 **Trading Education Center**\n\n*Learn professional trading:*\n\n📚 **Basics** - Fundamental concepts\n🎯 **Risk Management** - Protect your capital\n🤖 **AI Strategies** - How our AI works\n📊 **Technical Analysis** - Chart reading",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    async def handle_signal_request(self, query, signal_type):
        """Handle signal generation requests"""
        signal_names = {
            "signal_quick": "⚡ Quick Signal",
            "signal_trend": "📈 Trend Analysis", 
            "signal_pattern": "🎯 Pattern Scanner",
            "signal_volume": "📊 Volume Analysis"
        }
        
        signal_name = signal_names.get(signal_type, "AI Signal")
        
        # Simulate AI processing
        processing_msg = await query.edit_message_text(
            f"🔄 **{signal_name}**\n\n*AI is analyzing market data...*\n\n⏳ Please wait 5-10 seconds",
            parse_mode="Markdown"
        )
        
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Generate sample signal (will be replaced with real AI later)
        sample_signal = f"""
🎯 **{signal_name} - GENERATED**

📊 **EUR/USD - 5 MINUTE CALL**
✅ **Confidence:** 78%
🕒 **Expiry:** 5 minutes
🎯 **Entry Zone:** 1.0850-1.0860
📈 **Target:** 1.0880-1.0890
🛑 **Stop Loss:** Below 1.0830

💡 **AI Analysis:**
• Bullish trend detected
• RSI showing momentum
• Volume confirmation

⚡ **Action:** CALL position recommended
💰 **Payout:** 75-80% expected"""

        keyboard = [
            [InlineKeyboardButton("🔄 New Signal", callback_data="menu_signals")],
            [InlineKeyboardButton("📊 Different Asset", callback_data="signal_assets")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="menu_main")]
        ]
        
        await query.edit_message_text(
            sample_signal,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    async def handle_strategy_request(self, query, strategy_type):
        """Handle strategy information requests"""
        strategy_info = {
            "strategy_trend": "🚀 **Trend Spotter Strategy**\n\nFollows market trends using moving averages and momentum indicators.",
            "strategy_scalper": "⚡ **Scalper Pro Strategy**\n\nQuick 1-3 minute trades based on price action and volume.",
            "strategy_volume": "📊 **Volume Analysis Strategy**\n\nUses volume spikes and liquidity for high-probability entries.",
            "strategy_pattern": "🎯 **Pattern Master Strategy**\n\nAI pattern recognition for chart patterns and formations."
        }
        
        info = strategy_info.get(strategy_type, "Strategy information coming soon.")
        
        keyboard = [
            [InlineKeyboardButton("🎯 Use This Strategy", callback_data=f"signal_{strategy_type.split('_')[1]}")],
            [InlineKeyboardButton("📊 All Strategies", callback_data="menu_strategies")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="menu_main")]
        ]
        
        await query.edit_message_text(
            info,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    async def set_webhook(self):
        """Set webhook for Telegram"""
        await self.application.bot.set_webhook(self.webhook_url)
        logger.info(f"✅ Webhook set: {self.webhook_url}")

    async def process_update(self, update_data):
        """Process incoming update"""
        update = Update.de_json(update_data, self.application.bot)
        await self.application.process_update(update)
