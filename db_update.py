#!/usr/bin/env python3
"""
One-time database migration script for ModelHub CLI
Adds deploy_targets, deploy_mappings, and deploy_links tables to existing modelhub.db
"""

import sqlite3
import os
import sys
from pathlib import Path

def get_db_path():
    """Get the path to the modelhub.db file"""
    current_dir = Path(__file__).parent
    db_path = current_dir / "modelhub.db"
    return db_path

def check_existing_tables(cursor):
    """Check which tables already exist"""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}
    return existing_tables

def create_deploy_tables(cursor):
    """Create the new deploy-related tables"""
    
    # Create deploy_targets table
    cursor.execute("""
        CREATE TABLE deploy_targets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            base_path TEXT NOT NULL,
            enabled BOOLEAN DEFAULT TRUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✓ Created deploy_targets table")
    
    # Create deploy_mappings table
    cursor.execute("""
        CREATE TABLE deploy_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_id INTEGER NOT NULL,
            model_type TEXT NOT NULL,
            folder_path TEXT NOT NULL,
            FOREIGN KEY (target_id) REFERENCES deploy_targets (id) ON DELETE CASCADE,
            UNIQUE(target_id, model_type)
        )
    """)
    print("✓ Created deploy_mappings table")
    
    # Create deploy_links table
    cursor.execute("""
        CREATE TABLE deploy_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            source_path TEXT NOT NULL,
            deploy_path TEXT NOT NULL,
            is_deployed BOOLEAN DEFAULT FALSE,
            deployed_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES deploy_targets (id) ON DELETE CASCADE,
            UNIQUE(model_id, target_id)
        )
    """)
    print("✓ Created deploy_links table")
    
    # Create indexes for better performance
    cursor.execute("CREATE INDEX idx_deploy_targets_name ON deploy_targets (name)")
    cursor.execute("CREATE INDEX idx_deploy_mappings_target ON deploy_mappings (target_id)")
    cursor.execute("CREATE INDEX idx_deploy_mappings_type ON deploy_mappings (model_type)")
    cursor.execute("CREATE INDEX idx_deploy_links_model ON deploy_links (model_id)")
    cursor.execute("CREATE INDEX idx_deploy_links_target ON deploy_links (target_id)")
    cursor.execute("CREATE INDEX idx_deploy_links_deployed ON deploy_links (is_deployed)")
    print("✓ Created indexes")

def main():
    """Main migration function"""
    db_path = get_db_path()
    
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}")
        print("Please make sure you're running this script from the modelhub directory")
        sys.exit(1)
    
    print(f"Found database at: {db_path}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check existing tables
        existing_tables = check_existing_tables(cursor)
        print(f"Existing tables: {', '.join(sorted(existing_tables))}")
        
        # Check if we need to run migration
        deploy_tables = {'deploy_targets', 'deploy_mappings', 'deploy_links'}
        if deploy_tables.issubset(existing_tables):
            print("Deploy tables already exist. Migration not needed.")
            conn.close()
            return
        
        # Check if any deploy tables partially exist
        partial_tables = deploy_tables.intersection(existing_tables)
        if partial_tables:
            print(f"Warning: Some deploy tables already exist: {', '.join(partial_tables)}")
            response = input("Continue with migration? This might cause errors. (y/N): ")
            if response.lower() != 'y':
                print("Migration cancelled.")
                conn.close()
                return
        
        print("\nStarting database migration...")
        
        # Create new tables
        create_deploy_tables(cursor)
        
        # Commit changes
        conn.commit()
        print("✓ Database migration completed successfully")
        
        # Close connection
        conn.close()
        
        # Import and run default data creation
        print("\nCreating default deploy configurations...")
        try:
            from create_default_deploy import create_default_deploy_data
            create_default_deploy_data(db_path)
            print("✓ Default deploy data created successfully")
        except ImportError:
            print("Warning: create_default_deploy.py not found. You'll need to run it separately.")
        except Exception as e:
            print(f"Error creating default data: {e}")
            print("You can run create_default_deploy.py separately later.")
        
        print("\n🎉 Database update completed!")
        print("Your modelhub.db now supports deployment configurations.")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()