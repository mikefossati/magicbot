#!/usr/bin/env python3
"""
Database setup script for MagicBot
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.database import init_database
from src.core.config import load_config


def main():
    """Initialize the database with required tables"""
    config = load_config()
    init_database(config.database.url)
    print("Database initialized successfully!")


if __name__ == "__main__":
    main()
