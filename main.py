#!/usr/bin/env python3
"""
Binary Options AI Pro - Main Entry Point
FIXED: Asyncio event loop issue
"""

import os
import sys
import asyncio

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main function with proper asyncio handling"""
    from src.bot.main import main
    
    # Proper asyncio run for the bot
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Bot error: {e}")

if __name__ == "__main__":
    main()
