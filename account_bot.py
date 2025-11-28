# 📄 account_bot.py (WITH BUTTONS)
from flask import Flask, request, jsonify
import os
import requests
from datetime import datetime

app = Flask(__name__)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

user_limits = {}

@app.route('/account_webhook', methods=['POST'])
def account_webhook():
    update = request.get_json()
    
    if 'message' in update:
        chat_id = update['message']['chat']['id']
        text = update['message'].get('text', '')
        
        if text == '/start':
            send_account_dashboard(chat_id)
        elif text == '/account':
            send_account_dashboard(chat_id)
            
    elif 'callback_query' in update:
        chat_id = update['callback_query']['message']['chat']['id']
        message_id = update['callback_query']['message']['message_id']
        data = update['callback_query']['data']
        
        if data == 'account_dashboard':
            send_account_dashboard(chat_id, message_id)
        elif data == 'account_limits':
            show_account_limits(chat_id, message_id)
        elif data == 'account_upgrade':
            show_upgrade_options(chat_id, message_id)
    
    return jsonify({"status": "ok"})

def send_account_dashboard(chat_id, message_id=None):
    stats = get_user_stats(chat_id)
    
    keyboard = {
        "inline_keyboard": [
            [{"text": "📊 ACCOUNT LIMITS", "callback_data": "account_limits"}],
            [{"text": "💎 UPGRADE ACCOUNT", "callback_data": "account_upgrade"}],
            [{"text": "🎯 BACK TO TRADING", "callback_data": "back_to_trading"}]
        ]
    }
    
    text = f"""
💼 **ACCOUNT DASHBOARD**

*Professional OTC Trading Account*

📊 **Account Type:** {stats['account_type']}
🎯 **Signals Today:** {stats['signals_today']}/{stats['daily_limit']}
📈 **Status:** {'🟢 ACTIVE' if stats['signals_today'] < stats['daily_limit'] else '🔴 LIMIT REACHED'}

**FREE ACCOUNT FEATURES:**
✓ 10 signals per day
✓ All 15 trading assets
✓ 8 AI analysis engines  
✓ Market session tracking
✓ Professional strategies

*Manage your account settings below*"""
    
    send_telegram_message(chat_id, text, keyboard, message_id)

def show_account_limits(chat_id, message_id=None):
    stats = get_user_stats(chat_id)
    
    keyboard = {
        "inline_keyboard": [
            [{"text": "💎 UPGRADE FOR MORE", "callback_data": "account_upgrade"}],
            [{"text": "🔙 DASHBOARD", "callback_data": "account_dashboard"}]
        ]
    }
    
    text = f"""
📊 **ACCOUNT LIMITS**

*Your current signal usage:*

🎯 **DAILY SIGNALS:** {stats['signals_today']}/{stats['daily_limit']}
🕒 **RESET TIME:** Midnight UTC
💼 **ACCOUNT TYPE:** {stats['account_type']}

**UPGRADE BENEFITS:**
• Basic ($19): 50 signals/day
• Pro ($49): Unlimited signals  
• Enterprise ($149): All features + API

*Contact @admin to upgrade*"""
    
    send_telegram_message(chat_id, text, keyboard, message_id)

def show_upgrade_options(chat_id, message_id=None):
    keyboard = {
        "inline_keyboard": [
            [{"text": "💚 BASIC - $19/mo", "callback_data": "upgrade_basic"}],
            [{"text": "💎 PRO - $49/mo", "callback_data": "upgrade_pro"}],
            [{"text": "👑 ENTERPRISE - $149/mo", "callback_data": "upgrade_enterprise"}],
            [{"text": "🔙 DASHBOARD", "callback_data": "account_dashboard"}]
        ]
    }
    
    text = """
💎 **ACCOUNT UPGRADE OPTIONS**

**💚 BASIC - $19/month:**
• 50 signals per day
• All 15 assets
• Advanced AI engines
• Priority support

**💎 PRO - $49/month:**
• Unlimited signals
• All features unlocked
• Premium AI engines
• Dedicated support

**👑 ENTERPRISE - $149/month:**
• Everything in Pro
• API access
• White label options
• Custom strategies

*Contact @admin to upgrade your account*"""
    
    send_telegram_message(chat_id, text, keyboard, message_id)

def get_user_stats(chat_id):
    today = datetime.now().date().isoformat()
    
    if chat_id in user_limits and user_limits[chat_id]['date'] == today:
        count = user_limits[chat_id]['count']
    else:
        count = 0
    
    return {
        'signals_today': count,
        'daily_limit': 10,
        'account_type': 'Free'
    }

def send_telegram_message(chat_id, text, reply_markup=None, message_id=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    
    if reply_markup:
        data["reply_markup"] = reply_markup
        
    if message_id:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
        data["message_id"] = message_id
    
    requests.post(url, json=data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
