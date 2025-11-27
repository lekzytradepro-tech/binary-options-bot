#!/usr/bin/env python3
"""
Binary Options AI Pro - Main Entry Point
"""

import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def run_bot():
    from src.bot.main import main
    await main()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_bot())
