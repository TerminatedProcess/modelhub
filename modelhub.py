#!/usr/bin/env python3
"""
ModelHub CLI - Main Application Entry Point
AI Model Management with ncurses TUI
"""

import sys
import curses
import argparse
from pathlib import Path
from config import ConfigManager
from database import ModelHubDB
from tui import ModelHubTUI

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="ModelHub - AI Model Management CLI",
        prog="modelhub"
    )
    parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Configuration file path (default: config.yaml)"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version="ModelHub CLI 0.1.0"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.load_config()
        
        print(f"ModelHub CLI starting...")
        print(f"Config: {args.config}")
        print(f"Model Hub: {config.model_hub_path}")
        
        # Ensure model hub directory exists
        model_hub_path = config_manager.get_model_hub_path()
        model_hub_path.mkdir(parents=True, exist_ok=True)
        print(f"Model Hub Path: {model_hub_path}")
        
        # Initialize database
        db_path = config_manager.get_db_path()
        print(f"Database: {db_path}")
        
        # Database will be created automatically if it doesn't exist
        
        # Connect to database and verify tables
        with ModelHubDB(db_path) as db:
            try:
                # Test database connection by loading model count
                model_count = db.get_model_count()
                print(f"Found {model_count} models in database")
                
                # Test deploy targets
                deploy_targets = db.get_deploy_targets()
                print(f"Found {len(deploy_targets)} deploy targets")
                
            except Exception as e:
                print(f"Database error: {e}")
                print("Please ensure database is properly migrated with db_update.py")
                sys.exit(1)
        
        # Initialize and run TUI
        print("Starting TUI...")
        
        def run_tui(stdscr):
            with ModelHubDB(db_path) as db:
                tui = ModelHubTUI(config_manager, db)
                tui.run(stdscr)
        
        curses.wrapper(run_tui)
        
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()