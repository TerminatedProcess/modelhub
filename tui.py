#!/usr/bin/env python3
"""
Text User Interface for ModelHub CLI
Handles all ncurses interface operations
"""

import curses
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from database import ModelHubDB, Model, DeployTarget
from config import ConfigManager

class ModelHubTUI:
    """Main TUI application"""
    
    def __init__(self, config_manager: ConfigManager, database: ModelHubDB):
        self.config = config_manager
        self.db = database
        
        # Display state
        self.models: List[Model] = []
        self.model_types: List[tuple] = []
        self.deploy_targets: List[DeployTarget] = []
        
        # Navigation state
        self.current_page = 0
        self.selected_row = 0
        self.display_offset = 0  # For scrolling within the loaded models
        self.page_size = config_manager.config.page_size if config_manager.config else 20
        
        # Filter state
        self.current_filter = None
        self.search_term = None
        self.sort_by = "filename"
        self.sort_order = "ASC"
        
        # UI state
        self.status_message = ""
        self.show_help = False
        
        # Screen dimensions
        self.height = 0
        self.width = 0
        self.stdscr = None
    
    def run(self, stdscr):
        """Main application loop"""
        self.stdscr = stdscr
        self.setup_curses()
        self.load_initial_data()
        
        while True:
            self.draw_screen()
            key = stdscr.getch()
            
            if key == ord('q'):
                break
            elif key == curses.KEY_UP:
                self.move_selection(-1)
            elif key == curses.KEY_DOWN:
                self.move_selection(1)
            elif key == curses.KEY_LEFT:
                self.prev_page()
            elif key == curses.KEY_RIGHT:
                self.next_page()
            elif key == curses.KEY_PPAGE:  # Page Up
                self.page_up()
            elif key == curses.KEY_NPAGE:  # Page Down
                self.page_down()
            elif key == ord('d'):
                self.show_model_details()
            elif key == ord('f'):
                self.filter_models()
            elif key == ord('s'):
                self.search_models()
            elif key == ord('r'):
                self.reset_filters()
            elif key == ord('h'):
                self.toggle_help()
            elif key == ord('c'):
                self.configure_deploy()
            elif key == ord('D'):
                self.deploy_models()
            elif key == ord('e'):
                self.export_models()
            elif key == ord('C'):
                self.cleanup_menu()
            elif key == ord('S'):
                self.scan_directory()
            elif key == ord('R'):
                self.reclassify_models()
            elif key == ord('X'):
                self.delete_models()
            elif key == ord('\n') or key == curses.KEY_ENTER:
                self.show_model_details()
    
    def setup_curses(self):
        """Initialize curses settings"""
        curses.curs_set(0)  # Hide cursor
        self.height, self.width = self.stdscr.getmaxyx()
        
        # Initialize color pairs with readable combinations
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            
            # Color pairs: (foreground, background)
            curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLUE)   # Header - yellow on blue
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)    # Selected - black on cyan
            curses.init_pair(3, curses.COLOR_GREEN, -1)                   # Status - green on default
            curses.init_pair(4, curses.COLOR_YELLOW, -1)                  # Warning - yellow on default
            curses.init_pair(5, curses.COLOR_RED, -1)                     # Error - red on default
    
    def load_initial_data(self):
        """Load initial data from database"""
        try:
            self.load_models()
            self.load_model_types()
            self.load_deploy_targets()
            self.status_message = f"Loaded {len(self.models)} models"
        except Exception as e:
            self.status_message = f"Error loading data: {e}"
    
    def load_models(self):
        """Load models from database with current filters"""
        try:
            self.models = self.db.get_models(
                limit=self.page_size,
                offset=self.current_page * self.page_size,
                filter_type=self.current_filter,
                search_term=self.search_term,
                sort_by=self.sort_by,
                sort_order=self.sort_order
            )
        except Exception as e:
            self.status_message = f"Error loading models: {e}"
            self.models = []
    
    def load_model_types(self):
        """Load model types with counts"""
        try:
            self.model_types = self.db.get_model_types()
        except Exception as e:
            self.status_message = f"Error loading model types: {e}"
            self.model_types = []
    
    def load_deploy_targets(self):
        """Load deploy targets"""
        try:
            self.deploy_targets = self.db.get_deploy_targets()
        except Exception as e:
            self.status_message = f"Error loading deploy targets: {e}"
            self.deploy_targets = []
    
    def draw_screen(self):
        """Draw the main screen"""
        self.stdscr.clear()
        
        if self.show_help:
            self.draw_help_screen()
        else:
            self.draw_main_screen()
        
        self.stdscr.refresh()
    
    def draw_main_screen(self):
        """Draw the main application screen"""
        # Header
        header = "ModelHub - AI Model Management"
        self.stdscr.addstr(0, 0, header[:self.width-1], curses.color_pair(1))
        
        # Filter/search info
        filter_info = f"Filter: {self.current_filter or 'None'} | Search: {self.search_term or 'None'} | Page: {self.current_page + 1}"
        self.stdscr.addstr(1, 0, filter_info[:self.width-1])
        
        # Column headers
        headers = f"{'Model Name':<50} {'Type':<12} {'Subtype':<12} {'LoraTriggers':<30}"
        try:
            self.stdscr.addstr(3, 0, headers[:self.width-1], curses.A_BOLD)
        except curses.error:
            pass
        
        # Model list with scrolling
        max_display_rows = self.height - 7  # Reserve space for header, help, status
        
        for i in range(max_display_rows):
            model_index = self.display_offset + i
            if model_index >= len(self.models):
                break
                
            model = self.models[model_index]
            y = 4 + i
            
            try:
                # Show LoRA triggers only for LoRA models, otherwise blank
                triggers = ""
                if model.primary_type and 'lora' in model.primary_type.lower():
                    if model.triggers:
                        triggers = model.triggers[:30]  # Limit to 30 chars for display
                
                line = f"{model.filename[:50]:<50} {model.primary_type:<12} {model.sub_type:<12} {triggers:<30}"
                
                attr = curses.color_pair(2) if model_index == self.selected_row else 0
                self.stdscr.addstr(y, 0, line[:self.width-1], attr)
            except curses.error:
                pass
        
        # Help line
        help_line = "↑/↓ Select | PgUp/PgDn Jump | ←/→ Page | d Details | S Scan | f Filter | s Search | r Reset | h Help | q Quit"
        try:
            self.stdscr.addstr(self.height-2, 0, help_line[:self.width-1], curses.color_pair(3))
        except curses.error:
            pass
        
        # Status line
        if self.status_message:
            try:
                self.stdscr.addstr(self.height-1, 0, self.status_message[:self.width-1], curses.color_pair(3))
            except curses.error:
                pass
    
    def draw_help_screen(self):
        """Draw the help screen"""
        help_text = [
            "ModelHub CLI - Help",
            "",
            "Navigation:",
            "  ↑/↓ - Move selection up/down",
            "  PgUp/PgDn - Jump by page",
            "  ←/→ - Previous/next page",
            "  ENTER or 'd' - Show model details",
            "",
            "Filtering & Search:",
            "  'f' - Filter by model type",
            "  's' - Search in filenames/triggers",
            "  'r' - Reset all filters",
            "",
            "Model Management:",
            "  'S' - Scan directory for models",
            "  'R' - Reclassify models",
            "  'X' - Delete selected models",
            "",
            "Deployment:",
            "  'c' - Configure deploy targets",
            "  'D' - Deploy models",
            "  'e' - Export symlink commands",
            "",
            "Cleanup:",
            "  'C' - Cleanup menu",
            "",
            "Other:",
            "  'h' - Toggle this help",
            "  'q' - Quit application",
            "",
            "Press 'h' to return to main view"
        ]
        
        for i, line in enumerate(help_text):
            y = 2 + i
            if y >= self.height - 2:
                break
            try:
                self.stdscr.addstr(y, 2, line[:self.width-4])
            except curses.error:
                pass
    
    def move_selection(self, direction: int):
        """Move selection up or down with scrolling"""
        new_selection = self.selected_row + direction
        if 0 <= new_selection < len(self.models):
            self.selected_row = new_selection
            
            # Adjust display offset for scrolling
            max_display_rows = self.height - 7
            
            # Scroll down if selection is below visible area
            if self.selected_row >= self.display_offset + max_display_rows:
                self.display_offset = self.selected_row - max_display_rows + 1
                
            # Scroll up if selection is above visible area
            elif self.selected_row < self.display_offset:
                self.display_offset = self.selected_row
    
    def next_page(self):
        """Go to next page"""
        if len(self.models) == self.page_size:  # Might have more pages
            self.current_page += 1
            self.selected_row = 0
            self.load_models()
    
    def prev_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self.selected_row = 0
            self.display_offset = 0
            self.load_models()
    
    def page_up(self):
        """Jump up by visible page size"""
        max_display_rows = self.height - 7
        new_selection = max(0, self.selected_row - max_display_rows)
        self.selected_row = new_selection
        
        # Adjust display offset
        if self.selected_row < self.display_offset:
            self.display_offset = max(0, self.selected_row - max_display_rows // 2)
    
    def page_down(self):
        """Jump down by visible page size"""
        max_display_rows = self.height - 7
        new_selection = min(len(self.models) - 1, self.selected_row + max_display_rows)
        self.selected_row = new_selection
        
        # Adjust display offset
        if self.selected_row >= self.display_offset + max_display_rows:
            self.display_offset = min(len(self.models) - max_display_rows, 
                                    self.selected_row - max_display_rows // 2)
    
    def toggle_help(self):
        """Toggle help screen"""
        self.show_help = not self.show_help
    
    # Filter and search methods
    def filter_models(self):
        """Show filter dialog (STUB)"""
        self.status_message = "Filter functionality - STUB"
        # TODO: Implement filter dialog
    
    def search_models(self):
        """Show search dialog (STUB)"""
        self.status_message = "Search functionality - STUB"
        # TODO: Implement search dialog
    
    def reset_filters(self):
        """Reset all filters"""
        self.current_filter = None
        self.search_term = None
        self.current_page = 0
        self.selected_row = 0
        self.display_offset = 0
        self.load_models()
        self.status_message = "Filters reset"
    
    # Model operations (STUBS)
    def show_model_details(self):
        """Show detailed model information (STUB)"""
        if self.models and self.selected_row < len(self.models):
            model = self.models[self.selected_row]
            self.status_message = f"Show details for: {model.filename} - STUB"
        # TODO: Implement model details dialog
    
    def scan_directory(self):
        """Scan directory for new models - exit to console like pacman"""
        curses.curs_set(1)  # Show cursor
        curses.echo()
        
        # Get current working directory as default
        default_dir = os.getcwd()
        
        # Create input dialog
        scan_win = curses.newwin(7, 80, self.height//2-3, self.width//2-40)
        scan_win.box()
        scan_win.addstr(1, 2, "Scan Directory for Models")
        scan_win.addstr(3, 2, f"Directory [{default_dir}]: ")
        scan_win.refresh()
        
        try:
            # Get directory input
            input_str = scan_win.getstr(3, len(f"Directory [{default_dir}]: ") + 2, 60).decode('utf-8').strip()
            
            curses.noecho()
            curses.curs_set(0)
            del scan_win
            
            # If user didn't enter anything, cancel scan
            if not input_str:
                self.status_message = "Scan cancelled"
                return
            
            scan_dir = Path(input_str).expanduser().resolve()
            
            # Exit ncurses mode and go to console (like pacman)
            curses.endwin()
            
            try:
                print(f"\n=== ModelHub Scan ===")
                print(f"Scanning directory: {scan_dir}")
                print(f"Looking for extensions: {', '.join(self.config.config.file_extensions)}")
                print()
                
                # Get file extensions from config
                extensions = self.config.config.file_extensions
                
                # Scan for model files
                print("Finding model files...")
                model_files = self.db.scan_directory(scan_dir, extensions)
                
                if not model_files:
                    print("No model files found.")
                    input("\nPress Enter to continue...")
                    self.status_message = "No model files found"
                    return
                
                print(f"Found {len(model_files)} model files")
                print("\nImporting models...")
                print("-" * 60)
                
                # Import each model
                imported_count = 0
                skipped_count = 0
                model_hub_path = self.config.get_model_hub_path()
                
                for i, file_path in enumerate(model_files):
                    print(f"[{i+1}/{len(model_files)}] {file_path.name}")
                    
                    try:
                        model = self.db.import_model(file_path, model_hub_path, quiet=False, config_manager=self.config)
                        if model:
                            # Check if this was a new import (not existing)
                            if model.filename == file_path.name:
                                imported_count += 1
                                print(f"  ✓ Imported")
                            else:
                                skipped_count += 1
                                print(f"  ⚬ Already exists")
                    except Exception as e:
                        print(f"  ✗ Error: {e}")
                        skipped_count += 1
                    print()
                
                # Show results
                print("-" * 60)
                print(f"Scan complete!")
                print(f"  Imported: {imported_count} models")
                print(f"  Skipped:  {skipped_count} models")
                print()
                
                # Reload models to show new data
                self.current_page = 0
                self.selected_row = 0
                self.load_models()
                
                self.status_message = f"Scan complete: {imported_count} imported, {skipped_count} skipped"
                
                input("Press Enter to return to ModelHub...")
                
            except Exception as e:
                print(f"Error during scan: {e}")
                input("Press Enter to continue...")
                self.status_message = f"Scan error: {e}"
            
            finally:
                # Restart ncurses mode
                self.stdscr.clear()
                self.stdscr.refresh()
            
        except Exception as e:
            curses.noecho()
            curses.curs_set(0)
            if 'scan_win' in locals():
                del scan_win
            self.status_message = f"Scan error: {e}"
    
    def reclassify_models(self):
        """Reclassify models (STUB)"""
        self.status_message = "Reclassify functionality - STUB"
        # TODO: Implement reclassification
    
    def delete_models(self):
        """Delete selected models (STUB)"""
        self.status_message = "Delete functionality - STUB"
        # TODO: Implement model deletion
    
    # Deploy operations (STUBS)
    def configure_deploy(self):
        """Configure deployment targets (STUB)"""
        self.status_message = "Configure deploy targets - STUB"
        # TODO: Implement deploy configuration
    
    def deploy_models(self):
        """Deploy models to targets (STUB)"""
        self.status_message = "Deploy models functionality - STUB"
        # TODO: Implement model deployment
    
    def export_models(self):
        """Export symlink commands (STUB)"""
        self.status_message = "Export symlinks functionality - STUB"
        # TODO: Implement symlink export
    
    # Cleanup operations (STUB)
    def cleanup_menu(self):
        """Show cleanup menu (STUB)"""
        self.status_message = "Cleanup menu - STUB"
        # TODO: Implement cleanup menu