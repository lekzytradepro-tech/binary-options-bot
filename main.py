#!/usr/bin/env python3
"""
Entry point for local development
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def run_bot():
    from src.bot.main import main
    await main()

if __name__ == "__main__":
    asyncio.run(run_bot())
