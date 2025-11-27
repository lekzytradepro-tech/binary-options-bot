#!/usr/bin/env python3
"""
Binary Options AI Pro - Render Compatible Version
"""

import os
import sys
import asyncio

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_bot():
    """Simple run function for Render"""
    from src.bot.main import main
    asyncio.run(main())

if __name__ == "__main__":
    run_bot()
