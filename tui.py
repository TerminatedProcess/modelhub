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
        
        # Column positions for click detection (accounting for separators)
        self.column_positions = {
            'filename': (0, 45),
            'primary_type': (48, 60),
            'sub_type': (63, 83),
            'triggers': (86, 121),
            'classification_method': (124, 146)
        }
        
        # UI state
        self.status_message = ""
        self.show_help = False
        self.symlinks_enabled = True  # Default to symlinks enabled
        
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
                self.show_deploy_menu()
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
            elif key == ord('H'):
                self.generate_hash_files()
            elif key == ord('Y'):  # Shift+Y to toggle symlinks
                self.toggle_symlinks()
            elif key == ord('\n') or key == curses.KEY_ENTER:
                self.show_model_options()
    
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
            curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_GREEN)   # Selected - white on green
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
                # Always exclude deleted records
                all_conditions = ['deleted = 0'] + search_terms
                query = f"""
                SELECT id, file_hash, filename, file_size, file_extension,
                       primary_type, sub_type, confidence, classification_method,
                       tensor_count, architecture, precision, quantization,
                       triggers, filename_score, size_score, metadata_score,
                       tensor_score, classified_at, created_at, updated_at, reclassify, deleted
                FROM models
                WHERE {' AND '.join(all_conditions)}
                ORDER BY {self.get_sort_expression()} {self.sort_order}
                LIMIT {self.page_size} OFFSET {self.current_page * self.page_size}
                """
                
                cursor = self.db.conn.execute(query)
                self.models = []
                for row in cursor.fetchall():
                    from database import Model
                    self.models.append(Model(**dict(row)))
            else:
                # Use custom sorting for non-filtered queries too
                query = f"""
                SELECT id, file_hash, filename, file_size, file_extension,
                       primary_type, sub_type, confidence, classification_method,
                       tensor_count, architecture, precision, quantization,
                       triggers, filename_score, size_score, metadata_score,
                       tensor_score, classified_at, created_at, updated_at, reclassify, deleted
                FROM models
                WHERE deleted = 0
                ORDER BY {self.get_sort_expression()} {self.sort_order}
                LIMIT {self.page_size} OFFSET {self.current_page * self.page_size}
                """
                
                cursor = self.db.conn.execute(query)
                self.models = []
                for row in cursor.fetchall():
                    from database import Model
                    self.models.append(Model(**dict(row)))
        except Exception as e:
            self.status_message = f"Error loading models: {e}"
            self.models = []
    
    def get_sort_expression(self):
        """Get the SQL sort expression, handling case-insensitive sorting for text fields"""
        if self.sort_by == 'filename':
            return "LOWER(filename) COLLATE NOCASE"
        elif self.sort_by == 'primary_type':
            return "LOWER(primary_type) COLLATE NOCASE"
        elif self.sort_by == 'sub_type':
            return "LOWER(sub_type) COLLATE NOCASE"
        elif self.sort_by == 'classification_method':
            return "LOWER(classification_method) COLLATE NOCASE"
        elif self.sort_by == 'triggers':
            return "LOWER(triggers) COLLATE NOCASE"
        else:
            # For numeric fields, use as-is
            return self.sort_by
    
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
        symlinks_status = "on" if self.symlinks_enabled else "off"
        status_info = f"Page: {self.current_page + 1} | Total: {total_models} | Showing: {filtered_models} | Symlinks: {symlinks_status}"
        self.stdscr.addstr(0, 0, status_info[:49])
        
        
        # Filter fields on right side
        self.draw_filter_fields()
        
        # Column headers with separators
        headers = f"{'Model Name':<45} │ {'Type':<12} │ {'Subtype':<20} │ {'Triggers':<35} │ {'Method':<22}"
        try:
            self.stdscr.addstr(4, 0, headers[:self.width-1], curses.A_BOLD)
        except curses.error:
            pass
            
        # Header separator line
        separator_line = "─" * min(146, self.width-1)
        try:
            self.stdscr.addstr(5, 0, separator_line)
        except curses.error:
            pass
        
        # Model list with scrolling
        max_display_rows = self.height - 9  # Reserve space for header, separator, filters, help, status
        
        for i in range(max_display_rows):
            model_index = self.display_offset + i
            if model_index >= len(self.models):
                break
                
            model = self.models[model_index]
            y = 6 + i
            
            try:
                # Show LoRA triggers only for LoRA models, otherwise blank
                triggers = ""
                if model.primary_type and 'lora' in model.primary_type.lower():
                    if model.triggers:
                        triggers = model.triggers[:35]  # Increased to 35 chars for display
                
                # Classification method, shortened for display
                method = model.classification_method[:22] if model.classification_method else ""
                
                line = f"{model.filename[:45]:<45} │ {model.primary_type[:12]:<12} │ {model.sub_type[:20]:<20} │ {triggers:<35} │ {method:<22}"
                
                attr = curses.color_pair(2) if model_index == self.selected_row else 0
                self.stdscr.addstr(y, 0, line[:self.width-1], attr)
            except curses.error:
                pass
        
        # Help line
        help_line = "↑/↓ Select | PgUp/PgDn Jump | Click/F Filter | n Non-CivitAI | r Reset | R Reclassify | H Hash | Y Symlinks | D Deploy | h Help | q Quit"
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
            "  'H' - Generate hash files",
            "  'Y' - Toggle symlinks (on/off)",
            "",
            "Deployment:",
            "  'c' - Configure deploy targets",
            "  'D' - Deploy menu (packages & symlinks)",
            "  'e' - Export symlink commands",
            "",
            "Cleanup:",
            "  'C' - Cleanup menu",
            "",
            "Other:",
            "  'h' - Toggle this help",
            "  'q' - Quit application",
            "",
            "Symlinks:",
            "  When enabled (default), scanned models replace originals with symlinks",
            "  When disabled, original files are preserved during scanning",
            "  Setting resets to 'on' each time the application starts",
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
            max_display_rows = self.height - 9
            
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
        max_display_rows = self.height - 9
        new_selection = max(0, self.selected_row - max_display_rows)
        self.selected_row = new_selection
        
        # Adjust display offset
        if self.selected_row < self.display_offset:
            self.display_offset = max(0, self.selected_row - max_display_rows // 2)
    
    def page_down(self):
        """Jump down by visible page size"""
        max_display_rows = self.height - 9
        new_selection = min(len(self.models) - 1, self.selected_row + max_display_rows)
        self.selected_row = new_selection
        
        # Adjust display offset
        if self.selected_row >= self.display_offset + max_display_rows:
            self.display_offset = min(len(self.models) - max_display_rows, 
                                    self.selected_row - max_display_rows // 2)
    
    def toggle_help(self):
        """Toggle help screen"""
        self.show_help = not self.show_help
    
    def toggle_symlinks(self):
        """Toggle symlinks enabled/disabled"""
        self.symlinks_enabled = not self.symlinks_enabled
        status = "on" if self.symlinks_enabled else "off"
        self.status_message = f"Symlinks: {status}"
    
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
            elif my >= 6:  # Model list area
                self.deactivate_filter_field()
                # Calculate which model was clicked
                list_row = my - 6
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
            
            # Show sort indicator in status with model count
            direction = "↑" if self.sort_order == "ASC" else "↓"
            column_name = clicked_column.replace('_', ' ').title()
            self.status_message = f"Sorted by {column_name} {direction} ({len(self.models)} models)"
    
    # Model operations
    def show_model_options(self):
        """Show options popup for selected model"""
        if not self.models or self.selected_row >= len(self.models):
            self.status_message = "No model selected"
            return
            
        model = self.models[self.selected_row]
        
        # Create popup window
        popup_height = 8
        popup_width = 50
        popup_y = self.height // 2 - popup_height // 2
        popup_x = self.width // 2 - popup_width // 2
        
        popup_win = curses.newwin(popup_height, popup_width, popup_y, popup_x)
        popup_win.box()
        
        # Show model filename in title
        title = f" {model.filename[:44]} "
        popup_win.addstr(0, 2, title[:popup_width-4], curses.A_BOLD)
        
        # Menu options - reordered with hotkeys
        options = []
        
        # Add Copy Triggers as first option for LoRA models with triggers
        if model.primary_type and 'lora' in model.primary_type.lower() and model.triggers:
            options.append("Copy (t)riggers")
            
        options.extend([
            "(l)n -s to clipboard",
            "(r)e-classify",
            "(v)iew metadata",
            "(d)elete model"
        ])
        
        selected_option = 0
        
        while True:
            # Clear content area
            for i in range(1, popup_height - 1):
                popup_win.addstr(i, 1, " " * (popup_width - 2))
            
            # Draw options
            for i, option in enumerate(options):
                y = 2 + i
                attr = curses.color_pair(2) if i == selected_option else 0
                popup_win.addstr(y, 3, f"{option}", attr)
            
            popup_win.refresh()
            
            # Handle input
            key = self.stdscr.getch()  # Use main screen getch instead of popup
            
            if key == curses.KEY_UP and selected_option > 0:
                selected_option -= 1
            elif key == curses.KEY_DOWN and selected_option < len(options) - 1:
                selected_option += 1
            elif key == ord('\n') or key == curses.KEY_ENTER:
                del popup_win
                self.handle_model_option(model, selected_option)
                break
            elif key == 27 or key == ord('q'):  # Escape or 'q'
                del popup_win
                break
            # Handle hotkeys
            elif key == ord('t') or key == ord('T'):  # Copy triggers hotkey
                if model.primary_type and 'lora' in model.primary_type.lower() and model.triggers:
                    del popup_win
                    self.copy_model_triggers(model)
                    break
            elif key == ord('l') or key == ord('L'):  # ln -s to clipboard hotkey
                del popup_win
                self.copy_symlink_command(model)
                break
            elif key == ord('r') or key == ord('R'):  # Re-classify hotkey
                del popup_win
                self.reclassify_single_model(model)
                break
            elif key == ord('v') or key == ord('V'):  # View metadata hotkey
                del popup_win
                self.show_model_details(model)
                break
            elif key == ord('d') or key == ord('D'):  # Delete hotkey
                del popup_win
                self.delete_single_model(model)
                break
    
    def handle_model_option(self, model: Model, option_index: int):
        """Handle selected model option"""
        # Check if this is a LoRA with triggers (affects option indexing)
        has_copy_triggers = (model.primary_type and 'lora' in model.primary_type.lower() and model.triggers)
        
        if option_index == 0 and has_copy_triggers:  # Copy Triggers (LoRA only - first option)
            self.copy_model_triggers(model)
        elif option_index == 0 and not has_copy_triggers:  # ln -s to clipboard (first option for non-LoRA)
            self.copy_symlink_command(model)
        elif option_index == 1 and has_copy_triggers:  # ln -s to clipboard (second option for LoRA with triggers)
            self.copy_symlink_command(model)
        elif option_index == 1 and not has_copy_triggers:  # Re-classify (second option for non-LoRA)
            self.reclassify_single_model(model)
        elif option_index == 2 and has_copy_triggers:  # Re-classify (third option for LoRA with triggers)
            self.reclassify_single_model(model)
        elif option_index == 2 and not has_copy_triggers:  # View Metadata (third option for non-LoRA)
            self.show_model_details(model)
        elif option_index == 3 and has_copy_triggers:  # View Metadata (fourth option for LoRA with triggers)
            self.show_model_details(model)
        elif option_index == 3 and not has_copy_triggers:  # Delete (fourth option for non-LoRA)
            self.delete_single_model(model)
        elif option_index == 4:  # Delete (fifth option for LoRA with triggers)
            self.delete_single_model(model)
    
    def show_model_details(self, model: Model = None):
        """Show detailed model information"""
        try:
            if not model and self.models and self.selected_row < len(self.models):
                model = self.models[self.selected_row]
            
            if not model:
                self.status_message = "No model selected"
                return
            
            # Create scrollable details window (use most of the screen)
            popup_height = self.height - 2  # Leave just 1 line top and bottom
            popup_width = min(self.width - 4, 120)
            popup_y = 1
            popup_x = (self.width - popup_width) // 2
            
            popup_win = curses.newwin(popup_height, popup_width, popup_y, popup_x)
            popup_win.box()
            
            # Show title
            title = f" Model Details: {model.filename[:40]} "
            popup_win.addstr(0, 2, title, curses.A_BOLD)
            
            # Build details content
            details = self.build_model_details(model)
            
            # Scrolling state
            scroll_offset = 0
            max_content_lines = popup_height - 3  # Account for box and title
            
            while True:
                # Clear content area
                for i in range(1, popup_height - 1):
                    popup_win.addstr(i, 1, " " * (popup_width - 2))
                
                # Display details with scrolling
                for i in range(max_content_lines):
                    line_index = scroll_offset + i
                    if line_index >= len(details):
                        break
                    
                    line = details[line_index][:popup_width-4]  # Truncate if too long
                    try:
                        popup_win.addstr(i + 1, 2, line)
                    except curses.error:
                        pass
                
                # Show scroll indicator if needed
                if len(details) > max_content_lines:
                    total_lines = len(details)
                    visible_end = min(scroll_offset + max_content_lines, total_lines)
                    scroll_info = f" [{scroll_offset+1}-{visible_end}/{total_lines}] ↑/↓ Scroll | ESC/q Close "
                    try:
                        popup_win.addstr(popup_height-1, 2, scroll_info[:popup_width-4])
                    except curses.error:
                        pass
                else:
                    try:
                        popup_win.addstr(popup_height-1, 2, " Press ESC or 'q' to close ")
                    except curses.error:
                        pass
                
                popup_win.refresh()
                
                # Handle input
                key = self.stdscr.getch()
                
                if key == curses.KEY_UP and scroll_offset > 0:
                    scroll_offset -= 1
                elif key == curses.KEY_DOWN and scroll_offset + max_content_lines < len(details):
                    scroll_offset += 1
                elif key == curses.KEY_PPAGE and scroll_offset > 0:  # Page Up
                    scroll_offset = max(0, scroll_offset - max_content_lines)
                elif key == curses.KEY_NPAGE:  # Page Down
                    scroll_offset = min(len(details) - max_content_lines, scroll_offset + max_content_lines)
                elif key == 27 or key == ord('q'):  # Escape or 'q'
                    break
            
            del popup_win
            self.status_message = f"Viewed details for: {model.filename}"
            
        except Exception as e:
            self.status_message = f"Error showing model details: {e}"
    
    def build_model_details(self, model: Model) -> List[str]:
        """Build comprehensive model details as list of strings"""
        details = []
        
        # Basic Information
        details.append("═══ BASIC INFORMATION ═══")
        details.append(f"ID: {model.id}")
        details.append(f"Filename: {model.filename}")
        details.append(f"File Size: {model.file_size:,} bytes ({model.file_size / (1024*1024):.1f} MB)")
        details.append(f"Extension: {model.file_extension}")
        details.append(f"Hash: {model.file_hash}")
        details.append("")
        
        # Model Hub Path
        model_hub_path = self.config.get_model_hub_path()
        storage_path = model_hub_path / "models" / model.file_hash / model.filename
        details.append("═══ STORAGE LOCATION ═══")
        details.append(f"Hub Path: {storage_path}")
        details.append("")
        
        # Classification Results
        details.append("═══ CLASSIFICATION ═══")
        details.append(f"Primary Type: {model.primary_type}")
        details.append(f"Sub Type: {model.sub_type}")
        details.append(f"Confidence: {model.confidence:.3f}")
        details.append(f"Method: {model.classification_method}")
        details.append(f"Architecture: {model.architecture or 'Unknown'}")
        details.append("")
        
        # Technical Details
        details.append("═══ TECHNICAL DETAILS ═══")
        details.append(f"Tensor Count: {model.tensor_count or 'Unknown'}")
        details.append(f"Precision: {model.precision or 'Unknown'}")
        details.append(f"Quantization: {model.quantization or 'Unknown'}")
        details.append("")
        
        # Classification Scores
        details.append("═══ CLASSIFICATION SCORES ═══")
        details.append(f"Filename Score: {model.filename_score:.3f}")
        details.append(f"Size Score: {model.size_score:.3f}")
        details.append(f"Metadata Score: {model.metadata_score:.3f}")
        details.append(f"Tensor Score: {model.tensor_score:.3f}")
        details.append("")
        
        # Triggers (if LoRA)
        if model.triggers and 'lora' in model.primary_type.lower():
            details.append("═══ TRIGGER WORDS ═══")
            triggers = model.triggers.split(", ") if model.triggers else []
            for i, trigger in enumerate(triggers, 1):
                details.append(f"{i:2d}. {trigger}")
            details.append("")
        
        # Raw Metadata (comprehensive)
        try:
            metadata_dict = self.db.get_model_metadata_dict(model.id) 
            if metadata_dict:
                details.append("═══ EXTRACTED METADATA ═══")
                
                # CivitAI Response Data
                if 'civitai_response' in metadata_dict:
                    details.append("--- CivitAI Data ---")
                    import json
                    try:
                        civitai_data = json.loads(metadata_dict['civitai_response'])
                        details.append(f"Model Name: {civitai_data.get('name', 'Unknown')}")
                        details.append(f"CivitAI ID: {civitai_data.get('civitai_id', 'N/A')}")
                        details.append(f"Version ID: {civitai_data.get('version_id', 'N/A')}")
                        details.append(f"Base Model: {civitai_data.get('base_model', 'Unknown')}")
                        details.append(f"Source: {civitai_data.get('source', 'Unknown')}")
                        if civitai_data.get('triggers'):
                            details.append(f"CivitAI Triggers: {', '.join(civitai_data['triggers'])}")
                    except:
                        details.append("CivitAI data (parse error)")
                    details.append("")
                
                # SafeTensors Metadata
                if 'safetensors_metadata' in metadata_dict:
                    details.append("--- SafeTensors Metadata ---")
                    try:
                        st_data = json.loads(metadata_dict['safetensors_metadata'])
                        # Show key SafeTensors fields
                        important_keys = [
                            'ss_base_model_version', 'ss_network_module', 'ss_output_name',
                            'ss_tag_frequency', 'modelspec.architecture', 'modelspec.title',
                            'ss_training_comment', 'ss_dataset_dirs'
                        ]
                        for key in important_keys:
                            if key in st_data:
                                value = str(st_data[key])[:200]  # Truncate long values
                                details.append(f"{key}: {value}")
                        
                        # Show count of remaining fields
                        remaining_keys = [k for k in st_data.keys() if k not in important_keys]
                        if remaining_keys:
                            details.append(f"... and {len(remaining_keys)} more fields")
                    except:
                        details.append("SafeTensors metadata (parse error)")
                    details.append("")
                
                # Tensor Analysis
                if 'tensor_names' in metadata_dict:
                    details.append("--- Tensor Analysis ---")
                    try:
                        tensor_names = json.loads(metadata_dict['tensor_names'])
                        details.append(f"Total Tensors: {len(tensor_names)}")
                        # Show first few tensor names
                        for i, name in enumerate(tensor_names[:5]):
                            details.append(f"  {i+1}. {name}")
                        if len(tensor_names) > 5:
                            details.append(f"  ... and {len(tensor_names) - 5} more tensors")
                    except:
                        details.append("Tensor names (parse error)")
                    details.append("")
                
                # Tensor Scores
                if 'tensor_scores' in metadata_dict:
                    details.append("--- Tensor Pattern Scores ---")
                    try:
                        scores = json.loads(metadata_dict['tensor_scores'])
                        for pattern_type, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                            if score > 0:
                                details.append(f"{pattern_type}: {score:.3f}")
                    except:
                        details.append("Tensor scores (parse error)")
                    details.append("")
                
                # GGUF Metadata
                if 'gguf_metadata' in metadata_dict:
                    details.append("--- GGUF Metadata ---")
                    try:
                        gguf_data = json.loads(metadata_dict['gguf_metadata'])
                        details.append(f"GGUF Version: {gguf_data.get('version', 'Unknown')}")
                        details.append(f"Tensor Count: {gguf_data.get('tensor_count', 'Unknown')}")
                        details.append(f"Metadata Entries: {gguf_data.get('metadata_count', 'Unknown')}")
                    except:
                        details.append("GGUF metadata (parse error)")
                    details.append("")
                
                # File Information
                if 'file_size' in metadata_dict:
                    try:
                        file_size = int(metadata_dict['file_size'])
                        details.append("--- File Analysis ---")
                        details.append(f"Analyzed Size: {file_size:,} bytes ({file_size / (1024*1024):.1f} MB)")
                        if 'filename' in metadata_dict:
                            details.append(f"Original Filename: {metadata_dict['filename']}")
                        if 'file_extension' in metadata_dict:
                            details.append(f"Extension: {metadata_dict['file_extension']}")
                        details.append("")
                    except:
                        pass
                
            else:
                details.append("═══ NO METADATA AVAILABLE ═══")
                details.append("No raw metadata found for this model.")
                details.append("This may indicate the model was classified before")
                details.append("the enhanced metadata system was implemented.")
                details.append("")
                
        except Exception as e:
            details.append("═══ METADATA ERROR ═══")
            details.append(f"Error retrieving metadata: {e}")
            details.append("")
        
        return details
    
    def copy_model_triggers(self, model: Model):
        """Copy LoRA model triggers to clipboard"""
        if not model.triggers:
            self.status_message = f"No triggers found for {model.filename}"
            return
            
        try:
            from clipboard_utils import copy_to_clipboard
            
            # Clean up and format triggers
            triggers = model.triggers.strip()
            
            if copy_to_clipboard(triggers):
                self.status_message = f"Copied triggers to clipboard: {triggers}"
            else:
                self.status_message = f"Failed to copy triggers. Install xclip/wl-clipboard"
                
        except Exception as e:
            self.status_message = f"Clipboard error: {e}"
    
    def copy_symlink_command(self, model: Model):
        """Copy symbolic link command to clipboard"""
        try:
            from clipboard_utils import copy_to_clipboard
            
            # Get model hub path and construct full path to model file
            model_hub_path = self.config.get_model_hub_path()
            full_path = model_hub_path / "models" / model.file_hash / model.filename
            
            # Generate ln -s command
            symlink_command = f"ln -s {full_path} ."
            
            if copy_to_clipboard(symlink_command):
                self.status_message = f"Copied symlink command to clipboard: {model.filename}"
            else:
                self.status_message = f"Failed to copy symlink command. Install xclip/wl-clipboard"
                
        except Exception as e:
            self.status_message = f"Symlink clipboard error: {e}"
    
    def show_deploy_menu(self):
        """Show deployment options menu"""
        if not self.models:
            self.status_message = "No models loaded"
            return
            
        # Create popup window
        popup_height = 8
        popup_width = 50
        popup_y = self.height // 2 - popup_height // 2
        popup_x = self.width // 2 - popup_width // 2
        
        popup_win = curses.newwin(popup_height, popup_width, popup_y, popup_x)
        popup_win.box()
        
        # Show title
        title = " Deploy Options "
        popup_win.addstr(0, 2, title, curses.A_BOLD)
        
        # Menu options
        options = [
            "Deploy Packages",
            "Generate symbolic links",
            "Model names to clipboard"
        ]
        
        selected_option = 0
        
        while True:
            # Clear content area
            for i in range(1, popup_height - 1):
                popup_win.addstr(i, 1, " " * (popup_width - 2))
            
            # Draw options
            for i, option in enumerate(options):
                y = 2 + i
                attr = curses.color_pair(2) if i == selected_option else 0
                popup_win.addstr(y, 3, f"{option}", attr)
            
            popup_win.refresh()
            
            # Handle input
            key = self.stdscr.getch()
            
            if key == curses.KEY_UP and selected_option > 0:
                selected_option -= 1
            elif key == curses.KEY_DOWN and selected_option < len(options) - 1:
                selected_option += 1
            elif key == ord('\n') or key == curses.KEY_ENTER:
                del popup_win
                self.handle_deploy_option(selected_option)
                break
            elif key == 27 or key == ord('q'):  # Escape or 'q'
                del popup_win
                break
    
    def handle_deploy_option(self, option_index: int):
        """Handle selected deploy option"""
        if option_index == 0:  # Deploy Packages
            self.deploy_packages()
        elif option_index == 1:  # Generate symbolic links
            self.generate_symlinks_for_current_models()
        elif option_index == 2:  # Copy filenames list
            self.copy_filenames_list()
    
    def deploy_packages(self):
        """Deploy models to organized directories under model-hub/deploy/"""
        if not self.models:
            self.status_message = "No models to deploy"
            return
        
        try:
            # Get available deploy targets from database
            deploy_targets = self.db.get_deploy_targets()
            if not deploy_targets:
                self.status_message = "No deploy targets configured"
                return
            
            # Filter enabled targets
            enabled_targets = [target for target in deploy_targets if target.enabled]
            if not enabled_targets:
                self.status_message = "No deploy targets are enabled"
                return
            
            # Show deployment target selection popup
            selected_target = self.show_deploy_target_popup(enabled_targets)
            if selected_target is None:
                return  # User cancelled
            
            # Deploy to selected target
            self.deploy_to_target_package(selected_target)
            
        except Exception as e:
            self.status_message = f"Deployment failed: {e}"
    
    def generate_symlinks_for_current_models(self):
        """Generate symbolic link commands for all current filtered models"""
        if not self.models:
            self.status_message = "No models to generate links for"
            return
            
        try:
            from clipboard_utils import copy_to_clipboard
            
            # Get model hub path
            model_hub_path = self.config.get_model_hub_path()
            
            # Generate ln -s commands for all current models
            symlink_commands = []
            for model in self.models:
                full_path = model_hub_path / "models" / model.file_hash / model.filename
                symlink_commands.append(f"ln -s {full_path} .")
            
            # Join all commands with newlines
            commands_text = "\n".join(symlink_commands)
            
            if copy_to_clipboard(commands_text):
                self.status_message = f"Copied {len(symlink_commands)} symlink commands to clipboard"
            else:
                self.status_message = f"Failed to copy symlink commands. Install xclip/wl-clipboard"
                
        except Exception as e:
            self.status_message = f"Symlink generation error: {e}"
    
    def copy_filenames_list(self):
        """Copy list of filenames (no paths) for current filtered models"""
        if not self.models:
            self.status_message = "No models to copy filenames for"
            return
            
        try:
            from clipboard_utils import copy_to_clipboard
            
            # Generate list of just filenames
            filenames = [model.filename for model in self.models]
            
            # Join all filenames with newlines
            filenames_text = "\n".join(filenames)
            
            if copy_to_clipboard(filenames_text):
                self.status_message = f"Copied {len(filenames)} filenames to clipboard"
            else:
                self.status_message = f"Failed to copy filenames. Install xclip/wl-clipboard"
                
        except Exception as e:
            self.status_message = f"Filenames copy error: {e}"
    
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
                        # Pass the inverse of symlinks_enabled as preserve_originals
                        preserve_originals = not self.symlinks_enabled
                        model = self.db.import_model(file_path, model_hub_path, quiet=False, config_manager=self.config, preserve_originals=preserve_originals)
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
        """Reclassify models using current classification logic (respects current filters)"""
        # Exit ncurses mode for console output
        curses.endwin()
        
        try:
            print(f"\n=== ModelHub Reclassify ===")
            
            # Check config for reclassify behavior
            raw_config = self.config.get_raw_config()
            reclassify_civitai = raw_config.get('classification', {}).get('reclassify_civitai_models', False)
            
            # Build filter conditions based on current UI filters
            filter_conditions = ["deleted = 0"]  # Always exclude deleted records
            if self.model_filter:
                filter_conditions.append(f"LOWER(filename) LIKE '%{self.model_filter}%'")
            if self.type_filter:
                filter_conditions.append(f"LOWER(primary_type) LIKE '%{self.type_filter}%'")
            if self.subtype_filter:
                filter_conditions.append(f"LOWER(sub_type) LIKE '%{self.subtype_filter}%'")
            if self.non_civitai_filter:
                filter_conditions.append(f"classification_method != 'civitai_api'")
            
            # Add civitai preservation logic
            if not reclassify_civitai:
                filter_conditions.append("classification_method != 'civitai_api'")
            
            # Build where clause (always has conditions now due to deleted = 0)
            where_clause = "WHERE " + " AND ".join(filter_conditions)
            
            # Show what will be reclassified
            filter_desc = []
            if self.model_filter: filter_desc.append(f"model='{self.model_filter}'")
            if self.type_filter: filter_desc.append(f"type='{self.type_filter}'")
            if self.subtype_filter: filter_desc.append(f"sub='{self.subtype_filter}'")
            if self.non_civitai_filter: filter_desc.append("non-civitai")
            
            if filter_desc:
                print(f"Reclassifying filtered models: {', '.join(filter_desc)}")
            elif reclassify_civitai:
                print(f"Reclassifying ALL models using current logic...")
                print(f"(This will update trigger words for CivitAI models)")
            else:
                print(f"Reclassifying non-CivitAI models using current logic...")
                print(f"(CivitAI classifications are preserved as authoritative)")
            print()
            
            # Get models for reclassification based on current filters
            query = f"""
            SELECT id, file_hash, filename, file_size, file_extension,
                   primary_type, sub_type, confidence, classification_method,
                   tensor_count, architecture, precision, quantization,
                   triggers, filename_score, size_score, metadata_score,
                   tensor_score, classified_at, created_at, updated_at, reclassify, deleted
            FROM models
            {where_clause}
            ORDER BY filename ASC
            """
            cursor = self.db.conn.execute(query)
            all_models = []
            for row in cursor.fetchall():
                from database import Model
                all_models.append(Model(**dict(row)))
            
            if not all_models:
                print("No models found to reclassify with current filters.")
                input("\nPress Enter to continue...")
                return
            
            print(f"Found {len(all_models)} models to reclassify")
            print("-" * 60)
            
            # Load classifier with APIs disabled for speed during bulk reclassification
            from classifier import ModelClassifier
            raw_config = self.config.get_raw_config()
            # Temporarily disable external APIs for faster reclassification
            fast_config = raw_config.copy()
            fast_config['classification'] = fast_config.get('classification', {}).copy()
            fast_config['classification']['enable_external_apis'] = False
            classifier = ModelClassifier(fast_config, database=self.db)
            
            # Get model hub path to find the actual files
            model_hub_path = self.config.get_model_hub_path()
            
            reclassified_count = 0
            error_count = 0
            
            for i, model in enumerate(all_models):
                print(f"[{i+1}/{len(all_models)}] {model.filename}")
                
                try:
                    # Find the model file in storage
                    storage_path = model_hub_path / "models" / model.file_hash / model.filename
                    
                    if not storage_path.exists():
                        print(f"  ✗ File not found: {storage_path}")
                        error_count += 1
                        continue
                    
                    # Reclassify using current logic with existing hash (skip hash recalculation)
                    classification = classifier.classify_model(storage_path, model.file_hash, quiet=True, model_id=model.id)
                    
                    # Prepare trigger words for database storage
                    triggers_str = ", ".join(classification.triggers) if classification.triggers else None
                    
                    # Update database with new classification
                    from datetime import datetime
                    classified_at = datetime.now().isoformat()
                    
                    with self.db.conn:
                        self.db.conn.execute("""
                            UPDATE models SET 
                                primary_type = ?, sub_type = ?, confidence = ?, 
                                classification_method = ?, tensor_count = ?, 
                                architecture = ?, triggers = ?, classified_at = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (
                            classification.primary_type, classification.sub_type, 
                            classification.confidence, classification.method,
                            classification.tensor_count, classification.architecture,
                            triggers_str, classified_at, model.id
                        ))
                    
                    # Store raw metadata if available from fresh analysis
                    if hasattr(classification, 'raw_metadata') and classification.raw_metadata:
                        try:
                            self.db.store_model_metadata(model.id, classification.raw_metadata)
                            print(f"    📊 Stored fresh metadata ({len(classification.raw_metadata)} entries)")
                        except Exception as e:
                            print(f"    ⚠️  Warning: Failed to store metadata: {e}")
                    
                    print(f"  ✓ Reclassified: {classification.primary_type}/{classification.sub_type} ({classification.confidence:.2f})")
                    reclassified_count += 1
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    error_count += 1
            
            # Show results
            print("-" * 60)
            print(f"Reclassification complete!")
            print(f"  Reclassified: {reclassified_count} models")
            print(f"  Errors:       {error_count} models")
            print()
            
            # Reload models to show updated data
            self.current_page = 0
            self.selected_row = 0
            self.display_offset = 0
            
            self.status_message = f"Reclassified {reclassified_count} models, {error_count} errors"
            
            input("Press Enter to return to ModelHub...")
            
        except Exception as e:
            print(f"Error during reclassification: {e}")
            input("Press Enter to continue...")
            self.status_message = f"Reclassify error: {e}"
        
        finally:
            # Restart ncurses mode
            self.stdscr.clear()
            self.stdscr.refresh()
            # Reload the model list with updated data
            self.load_models()
    
    def reclassify_single_model(self, model: Model):
        """Reclassify a single model using current classification logic"""
        # Create progress popup
        progress_height = 5
        progress_width = 60
        progress_y = self.height // 2 - progress_height // 2
        progress_x = self.width // 2 - progress_width // 2
        
        progress_win = curses.newwin(progress_height, progress_width, progress_y, progress_x)
        progress_win.box()
        progress_win.addstr(1, 2, "Reclassifying model...", curses.A_BOLD)
        progress_win.addstr(2, 2, f"File: {model.filename[:52]}")
        progress_win.addstr(3, 2, "Please wait...")
        progress_win.refresh()
        
        try:
            # Load classifier with full configuration for accurate reclassification
            from classifier import ModelClassifier
            raw_config = self.config.get_raw_config()
            # Use full config including external APIs for best classification accuracy
            classifier = ModelClassifier(raw_config, database=self.db)
            
            # Get model hub path to find the actual file
            model_hub_path = self.config.get_model_hub_path()
            storage_path = model_hub_path / "models" / model.file_hash / model.filename
            
            if not storage_path.exists():
                # Show error message in the progress window
                progress_win.clear()
                progress_win.box()
                progress_win.addstr(1, 2, "Error!", curses.A_BOLD)
                progress_win.addstr(2, 2, f"File not found for {model.filename}")
                progress_win.addstr(3, 2, "Press any key to continue...")
                progress_win.refresh()
                
                # Wait for user input
                self.stdscr.getch()
                
                del progress_win
                self.status_message = f"Error: File not found for {model.filename}"
                return
            
            # Reclassify using current logic with existing hash (skip hash recalculation)
            classification = classifier.classify_model(storage_path, model.file_hash, quiet=True, model_id=model.id)
            
            # Prepare trigger words for database storage
            triggers_str = ", ".join(classification.triggers) if classification.triggers else None
            
            # Update database with new classification using transaction
            from datetime import datetime
            classified_at = datetime.now().isoformat()
            
            with self.db.conn:
                self.db.conn.execute("""
                    UPDATE models SET 
                        primary_type = ?, sub_type = ?, confidence = ?, 
                        classification_method = ?, tensor_count = ?, 
                        architecture = ?, triggers = ?, classified_at = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    classification.primary_type, classification.sub_type, 
                    classification.confidence, classification.method,
                    classification.tensor_count, classification.architecture,
                    triggers_str, classified_at, model.id
                ))
            
            # Store raw metadata if available from fresh analysis
            if hasattr(classification, 'raw_metadata') and classification.raw_metadata:
                try:
                    self.db.store_model_metadata(model.id, classification.raw_metadata)
                except Exception as e:
                    if not quiet:
                        print(f"Warning: Failed to store fresh metadata: {e}")
            
            # Show completion message in the progress window
            progress_win.clear()
            progress_win.box()
            progress_win.addstr(1, 2, "Reclassification Complete!", curses.A_BOLD)
            progress_win.addstr(2, 2, f"Result: {classification.primary_type}/{classification.sub_type}")
            progress_win.addstr(3, 2, f"Confidence: {classification.confidence:.2f}")
            progress_win.addstr(4, 2, "Press any key to continue...")
            progress_win.refresh()
            
            # Wait for user input
            self.stdscr.getch()
            
            # Close progress window
            del progress_win
            
            # Reload models to show updated data
            self.load_models()
            
            self.status_message = f"Reclassified: {model.filename} -> {classification.primary_type}/{classification.sub_type} ({classification.confidence:.2f})"
            
        except Exception as e:
            # Show error message in the progress window
            progress_win.clear()
            progress_win.box()
            progress_win.addstr(1, 2, "Error!", curses.A_BOLD)
            progress_win.addstr(2, 2, f"Error reclassifying {model.filename}")
            progress_win.addstr(3, 2, f"{str(e)[:50]}")
            progress_win.addstr(4, 2, "Press any key to continue...")
            progress_win.refresh()
            
            # Wait for user input
            self.stdscr.getch()
            
            del progress_win
            self.status_message = f"Error reclassifying {model.filename}: {e}"
    
    
    def delete_single_model(self, model: Model):
        """Soft delete a single model with confirmation"""
        # Show confirmation dialog
        if not self.show_delete_confirmation(f"Delete model: {model.filename}?"):
            return
        
        try:
            # Soft delete by setting deleted = 1
            self.db.conn.execute(
                "UPDATE models SET deleted = 1 WHERE id = ?",
                (model.id,)
            )
            self.db.conn.commit()
            
            # Reload models to remove deleted model from list
            self.load_models()
            
            self.status_message = f"Deleted {model.filename} (soft delete - use Shift-C cleanup to permanently remove)"
            
        except Exception as e:
            self.status_message = f"Error deleting {model.filename}: {e}"
    
    def delete_models(self):
        """Soft delete all filtered models with confirmation"""
        if not self.models:
            self.status_message = "No models to delete"
            return
        
        # Show confirmation dialog
        count = len(self.models)
        if not self.show_delete_confirmation(f"Delete {count} filtered models?"):
            return
        
        try:
            # Soft delete all current models by setting deleted = 1
            model_ids = [model.id for model in self.models]
            placeholders = ','.join(['?' for _ in model_ids])
            
            self.db.conn.execute(
                f"UPDATE models SET deleted = 1 WHERE id IN ({placeholders})",
                model_ids
            )
            self.db.conn.commit()
            
            # Reload models to remove deleted models from list
            self.load_models()
            
            self.status_message = f"Deleted {count} models (soft delete - use Shift-C cleanup to permanently remove)"
            
        except Exception as e:
            self.status_message = f"Error deleting models: {e}"
    
    def show_delete_confirmation(self, message):
        """Show confirmation dialog for delete operations"""
        popup_height = 7
        popup_width = max(len(message) + 10, 50)
        popup_y = self.height // 2 - popup_height // 2
        popup_x = self.width // 2 - popup_width // 2
        
        popup_win = curses.newwin(popup_height, popup_width, popup_y, popup_x)
        popup_win.box()
        
        # Show warning title
        title = " ⚠ CONFIRM DELETE ⚠ "
        popup_win.addstr(0, 2, title, curses.A_BOLD | curses.color_pair(1))
        
        # Show message
        popup_win.addstr(2, 3, message[:popup_width-6])
        
        # Show options
        popup_win.addstr(4, 3, "Press 'y' to confirm, any other key to cancel")
        
        popup_win.refresh()
        
        # Get user input
        key = self.stdscr.getch()
        del popup_win
        
        return key == ord('y') or key == ord('Y')
    
    def show_deploy_target_popup(self, enabled_targets):
        """Show popup to select deployment target"""
        popup_height = min(len(enabled_targets) + 5, 15)
        popup_width = 60
        popup_y = self.height // 2 - popup_height // 2
        popup_x = self.width // 2 - popup_width // 2
        
        popup_win = curses.newwin(popup_height, popup_width, popup_y, popup_x)
        popup_win.box()
        
        # Show title
        title = " Select Deployment Target "
        popup_win.addstr(0, 2, title, curses.A_BOLD)
        
        selected_option = 0
        
        while True:
            # Clear content area
            for i in range(1, popup_height - 1):
                popup_win.addstr(i, 1, " " * (popup_width - 2))
            
            # Draw options
            for i, target in enumerate(enabled_targets):
                y = 2 + i
                attr = curses.color_pair(2) if i == selected_option else 0
                display_name = target.display_name[:50]  # Truncate if too long
                popup_win.addstr(y, 3, f"{display_name}", attr)
            
            popup_win.refresh()
            
            # Handle input
            key = self.stdscr.getch()
            
            if key == curses.KEY_UP and selected_option > 0:
                selected_option -= 1
            elif key == curses.KEY_DOWN and selected_option < len(enabled_targets) - 1:
                selected_option += 1
            elif key == ord('\n') or key == curses.KEY_ENTER:
                del popup_win
                return enabled_targets[selected_option]
            elif key == 27 or key == ord('q'):  # Escape or 'q'
                del popup_win
                return None
    
    def deploy_to_target_package(self, target):
        """Deploy models to a specific target under model-hub/deploy/"""
        import shutil
        from pathlib import Path
        
        try:
            # Get model hub path
            model_hub_path = self.config.get_model_hub_path()
            
            # Create deployment directory structure: model-hub/deploy/target_name/
            deploy_base = model_hub_path / 'deploy' / target.name
            
            # Remove existing deployment directory and recreate fresh
            if deploy_base.exists():
                shutil.rmtree(deploy_base)
            
            # Create fresh deployment directory
            deploy_base.mkdir(parents=True, exist_ok=True)
            
            # Get deploy mappings for this target
            mappings = self.db.get_deploy_mappings(target.id)
            if not mappings:
                self.status_message = f"No deploy mappings configured for {target.display_name}"
                return
            
            # Create directories for each mapping
            created_dirs = set()
            for mapping in mappings:
                dir_path = deploy_base / mapping.folder_path
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.add(mapping.folder_path)
            
            # Always create unknown directory
            unknown_dir = deploy_base / 'unknown'
            unknown_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.add('unknown')
            
            # Build mapping dictionary for quick lookup
            type_to_folder = {}
            for mapping in mappings:
                type_to_folder[mapping.model_type] = mapping.folder_path
            
            # Deploy models
            deployed_count = 0
            skipped_count = 0
            error_count = 0
            skipped_models = []
            
            for model in self.models:
                try:
                    # Get the actual model file path in the hub
                    hub_model_path = model_hub_path / "models" / model.file_hash / model.filename
                    
                    # Check if model file exists in hub
                    if not hub_model_path.exists():
                        reason = "file not found in hub"
                        skipped_models.append((model.filename, reason))
                        skipped_count += 1
                        continue
                    
                    # Determine target directory
                    if model.primary_type in type_to_folder:
                        relative_dir = type_to_folder[model.primary_type]
                    elif model.primary_type == 'unknown' or model.primary_type is None:
                        relative_dir = 'unknown'
                    else:
                        # Skip unsupported model types
                        reason = f"no mapping for {model.primary_type}"
                        skipped_models.append((model.filename, reason))
                        skipped_count += 1
                        continue
                    
                    target_dir = deploy_base / relative_dir
                    
                    # Create symlink
                    symlink_path = target_dir / model.filename
                    
                    # Remove existing symlink if it exists
                    if symlink_path.exists() or symlink_path.is_symlink():
                        symlink_path.unlink()
                    
                    # Create new symlink to the hub model path
                    symlink_path.symlink_to(hub_model_path)
                    deployed_count += 1
                    
                except Exception as e:
                    reason = f"deployment error: {e}"
                    skipped_models.append((model.filename, reason))
                    error_count += 1
            
            # Set status message with summary
            status_parts = [f"Deployed {deployed_count} models to {target.display_name}"]
            if skipped_count > 0:
                status_parts.append(f"skipped {skipped_count}")
            if error_count > 0:
                status_parts.append(f"errors {error_count}")
            
            self.status_message = " | ".join(status_parts) + f" -> {deploy_base}"
            
        except Exception as e:
            self.status_message = f"Deployment to {target.display_name} failed: {e}"

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
    
    def generate_hash_files(self):
        """Generate hash files for all models in the database"""
        # Exit ncurses mode for console output
        curses.endwin()
        
        try:
            print(f"\n=== ModelHub Hash File Generation ===")
            print("Generating hash files for all models in the database...")
            
            # Get model hub path from config
            model_hub_path = self.config.get_model_hub_path()
            
            # Generate hash files
            files_processed, files_created = self.db.generate_missing_hash_files(model_hub_path, quiet=False)
            
            print(f"\nOperation complete!")
            print(f"Files processed: {files_processed}")
            print(f"Hash files created: {files_created}")
            
            # Set status message
            self.status_message = f"Generated {files_created} hash files ({files_processed} processed)"
            
            input("\nPress Enter to return to ModelHub...")
            
        except Exception as e:
            print(f"Error during hash file generation: {e}")
            input("Press Enter to continue...")
            self.status_message = f"Hash generation error: {e}"
        
        finally:
            # Restart ncurses mode
            self.stdscr.clear()
            self.stdscr.refresh()
    
    # Cleanup operations (STUB)
    def cleanup_menu(self):
        """Show cleanup options menu"""
        # Create popup window
        popup_height = 8
        popup_width = 50
        popup_y = self.height // 2 - popup_height // 2
        popup_x = self.width // 2 - popup_width // 2
        
        popup_win = curses.newwin(popup_height, popup_width, popup_y, popup_x)
        popup_win.box()
        
        # Show title
        title = " Cleanup Options "
        popup_win.addstr(0, 2, title, curses.A_BOLD)
        
        # Menu options
        options = [
            "(R)emove Deleted Models"
        ]
        
        selected_option = 0
        
        while True:
            # Clear content area
            for i in range(1, popup_height - 1):
                popup_win.addstr(i, 1, " " * (popup_width - 2))
            
            # Draw options
            for i, option in enumerate(options):
                y = 2 + i
                attr = curses.color_pair(2) if i == selected_option else 0
                popup_win.addstr(y, 3, f"{option}", attr)
            
            # Instructions
            popup_win.addstr(popup_height - 2, 2, "Enter to select, Esc/q to cancel")
            
            popup_win.refresh()
            
            # Handle input
            key = self.stdscr.getch()
            
            if key == curses.KEY_UP and selected_option > 0:
                selected_option -= 1
            elif key == curses.KEY_DOWN and selected_option < len(options) - 1:
                selected_option += 1
            elif key == ord('\n') or key == curses.KEY_ENTER:
                del popup_win
                self.handle_cleanup_option(selected_option)
                break
            elif key == ord('r') or key == ord('R'):
                del popup_win
                self.handle_cleanup_option(0)  # Remove deleted models
                break
            elif key == 27 or key == ord('q'):  # Escape or 'q'
                del popup_win
                break
    
    def handle_cleanup_option(self, option_index: int):
        """Handle selected cleanup option"""
        if option_index == 0:  # Remove Deleted Models
            self.remove_deleted_models()
    
    def remove_deleted_models(self):
        """Remove deleted models and cleanup orphaned files"""
        try:
            # Get model hub path
            model_hub_path = self.config.get_model_hub_path()
            
            # Perform cleanup operations
            result = self.db.cleanup_models(model_hub_path)
            
            # Display results
            deleted_count = result.get('deleted_records_removed', 0)
            orphaned_count = result.get('orphaned_records_removed', 0)
            reverse_orphan_count = result.get('reverse_orphans_moved', 0)
            
            message = f"Cleanup completed: {deleted_count} deleted records, {orphaned_count} orphaned records, {reverse_orphan_count} reverse orphans"
            self.status_message = message
            
            # Reload models to reflect changes
            self.load_models()
            
        except Exception as e:
            self.status_message = f"Cleanup failed: {str(e)}"