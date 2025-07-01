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
        self.model_filter = ""
        self.type_filter = ""
        self.subtype_filter = ""
        self.non_civitai_filter = False
        self.active_field = None  # None, 'model', 'type', 'subtype'
        self.sort_by = "filename"
        self.sort_order = "ASC"
        
        # Column positions for click detection
        self.column_positions = {
            'filename': (0, 50),
            'primary_type': (50, 62),
            'sub_type': (62, 77),
            'triggers': (77, 102),
            'classification_method': (102, 122)
        }
        
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
            elif key == curses.KEY_MOUSE:
                self.handle_mouse()
            elif self.active_field is not None:
                # Handle input for active filter field - this takes priority
                self.handle_filter_input(key)
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
            elif key == ord('F'):  # Shift+F to activate filter
                self.activate_filter_field('model')
            elif key == ord('r'):
                self.reset_filters()
            elif key == ord('n'):
                self.filter_non_civitai()
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
        curses.curs_set(0)  # Hide cursor initially
        self.height, self.width = self.stdscr.getmaxyx()
        
        # Enable mouse support
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
        
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
            curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLUE)    # Active field - white on blue
    
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
            # Build search term from filters (case-insensitive)
            search_terms = []
            if self.model_filter:
                search_terms.append(f"LOWER(filename) LIKE '%{self.model_filter}%'")
            if self.type_filter:
                search_terms.append(f"LOWER(primary_type) LIKE '%{self.type_filter}%'")
            if self.subtype_filter:
                search_terms.append(f"LOWER(sub_type) LIKE '%{self.subtype_filter}%'")
            if self.non_civitai_filter:
                search_terms.append(f"classification_method != 'civitai_api'")
            
            # Use raw SQL if we have filters, otherwise use existing method
            if search_terms:
                query = f"""
                SELECT id, file_hash, filename, file_size, file_extension,
                       primary_type, sub_type, confidence, classification_method,
                       tensor_count, architecture, precision, quantization,
                       triggers, filename_score, size_score, metadata_score,
                       tensor_score, classified_at, created_at, updated_at, reclassify
                FROM models
                WHERE {' AND '.join(search_terms)}
                ORDER BY {self.sort_by} {self.sort_order}
                LIMIT {self.page_size} OFFSET {self.current_page * self.page_size}
                """
                
                cursor = self.db.conn.execute(query)
                self.models = []
                for row in cursor.fetchall():
                    from database import Model
                    self.models.append(Model(**dict(row)))
            else:
                self.models = self.db.get_models(
                    limit=self.page_size,
                    offset=self.current_page * self.page_size,
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
        # Status info on left side
        total_models = self.get_total_model_count()
        filtered_models = len(self.models)
        status_info = f"Page: {self.current_page + 1} | Total: {total_models} | Showing: {filtered_models}"
        self.stdscr.addstr(0, 0, status_info[:49])
        
        
        # Filter fields on right side
        self.draw_filter_fields()
        
        # Column headers
        headers = f"{'Model Name':<50} {'Type':<12} {'Subtype':<15} {'LoraTriggers':<25} {'Method':<20}"
        try:
            self.stdscr.addstr(4, 0, headers[:self.width-1], curses.A_BOLD)
        except curses.error:
            pass
        
        # Model list with scrolling
        max_display_rows = self.height - 8  # Reserve space for header, filters, help, status
        
        for i in range(max_display_rows):
            model_index = self.display_offset + i
            if model_index >= len(self.models):
                break
                
            model = self.models[model_index]
            y = 5 + i
            
            try:
                # Show LoRA triggers only for LoRA models, otherwise blank
                triggers = ""
                if model.primary_type and 'lora' in model.primary_type.lower():
                    if model.triggers:
                        triggers = model.triggers[:25]  # Limit to 25 chars for display
                
                # Classification method, shortened for display
                method = model.classification_method[:20] if model.classification_method else ""
                
                line = f"{model.filename[:50]:<50} {model.primary_type:<12} {model.sub_type:<15} {triggers:<25} {method:<20}"
                
                attr = curses.color_pair(2) if model_index == self.selected_row else 0
                self.stdscr.addstr(y, 0, line[:self.width-1], attr)
            except curses.error:
                pass
        
        # Help line
        help_line = "↑/↓ Select | PgUp/PgDn Jump | Click/F Filter | n Non-CivitAI | r Reset | h Help | q Quit"
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
            "  Mouse click - Select model or activate filter",
            "",
            "Live Filtering:",
            "  Click on filter fields or press 'F' to start filtering",
            "  Tab - Move between Model/Type/SubType fields",
            "  Type to filter instantly (lowercase, partial match)",
            "  'n' - Toggle non-CivitAI models filter",
            "  Escape/Enter - Exit filter mode",
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
    
    def draw_filter_fields(self):
        """Draw the three filter input fields aligned with Type column"""
        type_col_start = 50  # Align with Type column
        field_width = min(25, self.width - type_col_start - 2)
        
        # Model filter field (line 0)
        model_attr = curses.color_pair(6) if self.active_field == 'model' else 0
        try:
            self.stdscr.addstr(0, type_col_start, "Model: ")
            filter_display = self.model_filter[:field_width-1] if self.model_filter else ""
            field_str = f"{filter_display:<{field_width}}"
            self.stdscr.addstr(0, type_col_start + 7, field_str, model_attr)
            if self.active_field == 'model':
                # Position cursor at end of text
                cursor_x = type_col_start + 7 + len(filter_display)
                self.stdscr.move(0, min(cursor_x, self.width-1))
        except curses.error:
            pass
            
        # Type filter field (line 1)
        type_attr = curses.color_pair(6) if self.active_field == 'type' else 0
        try:
            self.stdscr.addstr(1, type_col_start, "Type:  ")
            filter_display = self.type_filter[:field_width-1] if self.type_filter else ""
            field_str = f"{filter_display:<{field_width}}"
            self.stdscr.addstr(1, type_col_start + 7, field_str, type_attr)
            if self.active_field == 'type':
                # Position cursor at end of text
                cursor_x = type_col_start + 7 + len(filter_display)
                self.stdscr.move(1, min(cursor_x, self.width-1))
        except curses.error:
            pass
            
        # SubType filter field (line 2)
        subtype_attr = curses.color_pair(6) if self.active_field == 'subtype' else 0
        try:
            self.stdscr.addstr(2, type_col_start, "Sub:   ")
            filter_display = self.subtype_filter[:field_width-1] if self.subtype_filter else ""
            field_str = f"{filter_display:<{field_width}}"
            self.stdscr.addstr(2, type_col_start + 7, field_str, subtype_attr)
            if self.active_field == 'subtype':
                # Position cursor at end of text
                cursor_x = type_col_start + 7 + len(filter_display)
                self.stdscr.move(2, min(cursor_x, self.width-1))
        except curses.error:
            pass
    
    def get_total_model_count(self):
        """Get total model count without filters"""
        try:
            return self.db.get_model_count()
        except:
            return 0
    
    def move_selection(self, direction: int):
        """Move selection up or down with scrolling"""
        new_selection = self.selected_row + direction
        if 0 <= new_selection < len(self.models):
            self.selected_row = new_selection
            
            # Adjust display offset for scrolling
            max_display_rows = self.height - 8
            
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
        max_display_rows = self.height - 8
        new_selection = max(0, self.selected_row - max_display_rows)
        self.selected_row = new_selection
        
        # Adjust display offset
        if self.selected_row < self.display_offset:
            self.display_offset = max(0, self.selected_row - max_display_rows // 2)
    
    def page_down(self):
        """Jump down by visible page size"""
        max_display_rows = self.height - 8
        new_selection = min(len(self.models) - 1, self.selected_row + max_display_rows)
        self.selected_row = new_selection
        
        # Adjust display offset
        if self.selected_row >= self.display_offset + max_display_rows:
            self.display_offset = min(len(self.models) - max_display_rows, 
                                    self.selected_row - max_display_rows // 2)
    
    def toggle_help(self):
        """Toggle help screen"""
        self.show_help = not self.show_help
    
    def handle_mouse(self):
        """Handle mouse events"""
        try:
            _, mx, my, _, _ = curses.getmouse()
            
            
            # Check if click is on filter fields (aligned with Type column)
            if my == 0 and mx >= 57:  # Model filter line
                self.activate_filter_field('model')
            elif my == 1 and mx >= 57:  # Type filter line
                self.activate_filter_field('type')
            elif my == 2 and mx >= 57:  # SubType filter line
                self.activate_filter_field('subtype')
            elif my == 4:  # Column headers line
                self.handle_column_header_click(mx)
            elif my >= 5:  # Model list area
                self.deactivate_filter_field()
                # Calculate which model was clicked
                list_row = my - 5
                if list_row < len(self.models):
                    self.selected_row = self.display_offset + list_row
        except curses.error:
            pass
    
    def activate_filter_field(self, field):
        """Activate a filter field for editing"""
        self.active_field = field
        curses.curs_set(1)  # Show cursor
    
    def deactivate_filter_field(self):
        """Deactivate filter field editing"""
        self.active_field = None
        curses.curs_set(0)  # Hide cursor
    
    def handle_filter_input(self, key):
        """Handle input for active filter field"""
        if key == 27:  # Escape
            self.deactivate_filter_field()
        elif key == ord('\t'):  # Tab - move to next field
            if self.active_field == 'model':
                self.activate_filter_field('type')
            elif self.active_field == 'type':
                self.activate_filter_field('subtype')
            elif self.active_field == 'subtype':
                self.activate_filter_field('model')
        elif key == curses.KEY_ENTER or key == ord('\n'):
            self.deactivate_filter_field()
        elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
            # Backspace
            if self.active_field == 'model' and self.model_filter:
                self.model_filter = self.model_filter[:-1]
                self.apply_filters()
            elif self.active_field == 'type' and self.type_filter:
                self.type_filter = self.type_filter[:-1]
                self.apply_filters()
            elif self.active_field == 'subtype' and self.subtype_filter:
                self.subtype_filter = self.subtype_filter[:-1]
                self.apply_filters()
        elif 32 <= key <= 126:  # Printable characters
            char = chr(key).lower()  # Convert to lowercase
            if self.active_field == 'model':
                self.model_filter += char
                self.apply_filters()
            elif self.active_field == 'type':
                self.type_filter += char
                self.apply_filters()
            elif self.active_field == 'subtype':
                self.subtype_filter += char
                self.apply_filters()
    
    def apply_filters(self):
        """Apply current filters and reload models"""
        self.selected_row = 0
        self.display_offset = 0
        self.current_page = 0
        self.load_models()
        # Show current filter state
        filters = []
        if self.model_filter: filters.append(f"model='{self.model_filter}'")
        if self.type_filter: filters.append(f"type='{self.type_filter}'")
        if self.subtype_filter: filters.append(f"sub='{self.subtype_filter}'")
        if self.non_civitai_filter: filters.append("non-civitai")
        if filters:
            self.status_message = f"Filtering: {', '.join(filters)}"
        else:
            self.status_message = ""
    
    def reset_filters(self):
        """Reset all filters"""
        self.model_filter = ""
        self.type_filter = ""
        self.subtype_filter = ""
        self.non_civitai_filter = False
        self.active_field = None
        self.current_page = 0
        self.selected_row = 0
        self.display_offset = 0
        curses.curs_set(0)  # Hide cursor
        self.load_models()
        self.status_message = "Filters reset"
    
    def filter_non_civitai(self):
        """Filter to show only non-civitai classified models"""
        self.non_civitai_filter = not self.non_civitai_filter
        self.selected_row = 0
        self.display_offset = 0
        self.current_page = 0
        self.load_models()
    
    def handle_column_header_click(self, mx):
        """Handle clicks on column headers for sorting"""
        # Determine which column was clicked
        clicked_column = None
        for column, (start, end) in self.column_positions.items():
            if start <= mx < end:
                clicked_column = column
                break
        
        if clicked_column:
            # If clicking the same column, toggle sort order
            if self.sort_by == clicked_column:
                self.sort_order = "DESC" if self.sort_order == "ASC" else "ASC"
            else:
                # New column, default to ASC
                self.sort_by = clicked_column
                self.sort_order = "ASC"
            
            # Reset position and reload
            self.selected_row = 0
            self.display_offset = 0
            self.current_page = 0
            self.load_models()
            
            # Show sort indicator in status
            direction = "↑" if self.sort_order == "ASC" else "↓"
            column_name = clicked_column.replace('_', ' ').title()
            self.status_message = f"Sorted by {column_name} {direction}"
    
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