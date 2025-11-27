#!/usr/bin/env python3
"""
Local development entry point
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # For local development without web server
    from src.bot.main import main
    main()
