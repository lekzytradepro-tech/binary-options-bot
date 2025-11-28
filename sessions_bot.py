# 📄 sessions_bot.py (CREATE IN ROOT FOLDER)
from flask import Flask, request, jsonify
import os
import requests
from datetime import datetime

app = Flask(__name__)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

MARKET_SESSIONS = {
    "asian": {
        "name": "Asian Session",
        "time_utc": "22:00 - 06:00",
        "active_pairs": ["USD/JPY", "AUD/USD", "NZD/USD"],
        "volatility": "Low to Medium",
        "trading_tips": [
            "Trade USD/JPY and AUD/USD pairs",
            "Use mean reversion strategies", 
            "Focus on 15-30 minute expiries"
        ]
    },
    "london": {
        "name": "London Session", 
        "time_utc": "07:00 - 16:00",
        "active_pairs": ["EUR/USD", "GBP/USD", "EUR/GBP"],
        "volatility": "High",
        "trading_tips": [
            "Trade EUR/USD and GBP/USD pairs",
            "Use trend-following strategies",
            "Perfect for 5-15 minute expiries"
        ]
    },
    "new_york": {
        "name": "New York Session",
        "time_utc": "12:00 - 21:00", 
        "active_pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "US30"],
        "volatility": "Very High", 
        "trading_tips": [
            "Trade all major pairs",
            "Use momentum strategies",
            "Great for 1-5 minute scalping"
        ]
    }
}

@app.route('/sessions_webhook', methods=['POST'])
def sessions_webhook():
    update = request.get_json()
    
    if 'message' in update:
        chat_id = update['message']['chat']['id']
        text = update['message'].get('text', '')
        
        if text == '/start':
            send_sessions_dashboard(chat_id)
        elif text == '/sessions':
            send_sessions_dashboard(chat_id)
            
    elif 'callback_query' in update:
        chat_id = update['callback_query']['message']['chat']['id']
        message_id = update['callback_query']['message']['message_id']
        data = update['callback_query']['data']
        
        if data == 'sessions_dashboard':
            send_sessions_dashboard(chat_id, message_id)
        elif data == 'session_asian':
            show_session_detail(chat_id, message_id, 'asian')
        elif data == 'session_london':
            show_session_detail(chat_id, message_id, 'london')
        elif data == 'session_new_york':
            show_session_detail(chat_id, message_id, 'new_york')
        elif data == 'session_now':
            show_current_sessions(chat_id, message_id)
    
    return jsonify({"status": "ok"})

def send_sessions_dashboard(chat_id, message_id=None):
    current_time = datetime.utcnow().strftime("%H:%M UTC")
    
    keyboard = {
        "inline_keyboard": [
            [{"text": "🌏 ASIAN SESSION", "callback_data": "session_asian"}],
            [{"text": "🇬🇧 LONDON SESSION", "callback_data": "session_london"}],
            [{"text": "🇺🇸 NEW YORK SESSION", "callback_data": "session_new_york"}],
            [{"text": "🕒 ACTIVE NOW", "callback_data": "session_now"}],
            [{"text": "🎯 BACK TO TRADING", "callback_data": "back_to_trading"}]
        ]
    }
    
    text = f"""
🕒 **MARKET SESSIONS DASHBOARD**

*Professional Session Tracking for OTC Trading*

*Current Time: {current_time}*

**MAJOR TRADING SESSIONS:**
• 🌏 Asian - Low volatility, ranging markets
• 🇬🇧 London - High liquidity, strong trends  
• 🇺🇸 New York - Highest volatility, news driven

*Select a session for detailed analysis*"""
    
    send_telegram_message(chat_id, text, keyboard, message_id)

def show_session_detail(chat_id, message_id, session_id):
    session = MARKET_SESSIONS.get(session_id, {})
    current_time = datetime.utcnow().strftime("%H:%M UTC")
    
    keyboard = {
        "inline_keyboard": [
            [{"text": "🕒 ALL SESSIONS", "callback_data": "sessions_dashboard"}],
            [{"text": "🎯 BACK TO TRADING", "callback_data": "back_to_trading"}]
        ]
    }
    
    text = f"""
{session.get('name', 'Session')}

**🕒 SESSION TIMES:**
• UTC: {session.get('time_utc', 'N/A')}

**📊 MARKET CHARACTERISTICS:**
• Volatility: {session.get('volatility', 'N/A')}
• Best Pairs: {', '.join(session.get('active_pairs', []))}

**🎯 PROFESSIONAL TIPS:**
"""
    
    for tip in session.get('trading_tips', []):
        text += f"• {tip}\n"
    
    text += f"\n*Current Time: {current_time}*"
    
    send_telegram_message(chat_id, text, keyboard, message_id)

def show_current_sessions(chat_id, message_id):
    current_utc = datetime.utcnow()
    current_time = current_utc.strftime("%H:%M UTC")
    
    active_sessions = []
    for session_id, session in MARKET_SESSIONS.items():
        start_hour = int(session['time_utc'].split(' - ')[0].split(':')[0])
        end_hour = int(session['time_utc'].split(' - ')[1].split(':')[0])
        current_hour = current_utc.hour
        
        if start_hour > end_hour:  # Overnight
            if current_hour >= start_hour or current_hour < end_hour:
                active_sessions.append(session['name'])
        else:  # Day session
            if start_hour <= current_hour < end_hour:
                active_sessions.append(session['name'])
    
    active_text = "• " + "\n• ".join(active_sessions) if active_sessions else "• No active sessions"
    
    keyboard = {
        "inline_keyboard": [
            [{"text": "🕒 ALL SESSIONS", "callback_data": "sessions_dashboard"}],
            [{"text": "🎯 BACK TO TRADING", "callback_data": "back_to_trading"}]
        ]
    }
    
    text = f"""
🟢 **ACTIVE SESSIONS NOW**

*Current Time: {current_time}*

{active_text}

**TRADING RECOMMENDATION:**
Trade during active sessions for best liquidity and movement.

*Select a session for detailed analysis*"""
    
    send_telegram_message(chat_id, text, keyboard, message_id)

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
    app.run(host='0.0.0.0', port=5002)
