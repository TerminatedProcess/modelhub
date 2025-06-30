#!/usr/bin/env python3
"""
ModelHub CLI - ncurses interface for the modelhub SQLite database
"""

import curses
import sqlite3
import os
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Model:
    id: int
    filename: str
    primary_type: str
    sub_type: str
    confidence: float
    file_size: int
    architecture: Optional[str]
    precision: Optional[str]
    triggers: Optional[str]
    classified_at: str


class ModelHubDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.connect()
    
    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise Exception(f"Database connection failed: {e}")
    
    def get_models(self, limit: int = 100, offset: int = 0, 
                   filter_type: Optional[str] = None, 
                   search_term: Optional[str] = None) -> List[Model]:
        query = """
        SELECT id, filename, primary_type, sub_type, confidence, file_size,
               architecture, precision, triggers, classified_at
        FROM models
        """
        params = []
        
        conditions = []
        if filter_type:
            conditions.append("primary_type = ?")
            params.append(filter_type)
        
        if search_term:
            conditions.append("(filename LIKE ? OR triggers LIKE ?)")
            params.extend([f"%{search_term}%", f"%{search_term}%"])
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY classified_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = self.conn.execute(query, params)
        models = []
        for row in cursor.fetchall():
            models.append(Model(
                id=row['id'],
                filename=row['filename'],
                primary_type=row['primary_type'],
                sub_type=row['sub_type'],
                confidence=row['confidence'],
                file_size=row['file_size'],
                architecture=row['architecture'],
                precision=row['precision'],
                triggers=row['triggers'],
                classified_at=row['classified_at']
            ))
        return models
    
    def get_model_types(self) -> List[Tuple[str, int]]:
        cursor = self.conn.execute("""
            SELECT primary_type, COUNT(*) as count 
            FROM models 
            GROUP BY primary_type 
            ORDER BY count DESC
        """)
        return cursor.fetchall()
    
    def get_model_metadata(self, model_id: int) -> List[Tuple[str, str]]:
        cursor = self.conn.execute("""
            SELECT key, value 
            FROM model_metadata 
            WHERE model_id = ?
            ORDER BY key
        """, (model_id,))
        return cursor.fetchall()
    
    def close(self):
        if self.conn:
            self.conn.close()


class ModelHubCLI:
    def __init__(self, db_path: str):
        self.db = ModelHubDB(db_path)
        self.models = []
        self.model_types = []
        self.current_page = 0
        self.selected_row = 0
        self.page_size = 20
        self.current_filter = None
        self.search_term = None
        self.status_message = ""
        
    def run(self, stdscr):
        curses.curs_set(0)  # Hide cursor
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)   # Header
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Selected
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Status
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Warning
        
        self.stdscr = stdscr
        self.height, self.width = stdscr.getmaxyx()
        
        # Load initial data
        self.load_models()
        self.load_model_types()
        
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
            elif key == ord('d'):
                self.show_model_details()
            elif key == ord('f'):
                self.filter_models()
            elif key == ord('s'):
                self.search_models()
            elif key == ord('r'):
                self.reset_filters()
            elif key == ord('h'):
                self.show_help()
            elif key == ord('\n') or key == curses.KEY_ENTER:
                self.show_model_details()
    
    def load_models(self):
        try:
            self.models = self.db.get_models(
                limit=self.page_size,
                offset=self.current_page * self.page_size,
                filter_type=self.current_filter,
                search_term=self.search_term
            )
            self.status_message = f"Loaded {len(self.models)} models"
        except Exception as e:
            self.status_message = f"Error loading models: {e}"
    
    def load_model_types(self):
        try:
            self.model_types = self.db.get_model_types()
        except Exception as e:
            self.status_message = f"Error loading model types: {e}"
    
    def move_selection(self, direction: int):
        new_selection = self.selected_row + direction
        if 0 <= new_selection < len(self.models):
            self.selected_row = new_selection
    
    def next_page(self):
        if len(self.models) == self.page_size:  # Might have more pages
            self.current_page += 1
            self.selected_row = 0
            self.load_models()
    
    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.selected_row = 0
            self.load_models()
    
    def draw_screen(self):
        self.stdscr.clear()
        
        # Header
        header = "ModelHub CLI - Machine Learning Model Database Browser"
        self.stdscr.addstr(0, 0, header[:self.width-1], curses.color_pair(1))
        
        # Filter/search info
        filter_info = f"Filter: {self.current_filter or 'None'} | Search: {self.search_term or 'None'} | Page: {self.current_page + 1}"
        self.stdscr.addstr(1, 0, filter_info[:self.width-1])
        
        # Column headers
        headers = f"{'Type':<12} {'Filename':<40} {'Confidence':<10} {'Size':<10}"
        self.stdscr.addstr(3, 0, headers[:self.width-1], curses.A_BOLD)
        
        # Model list
        for i, model in enumerate(self.models):
            y = 4 + i
            if y >= self.height - 3:
                break
            
            size_mb = f"{model.file_size / (1024*1024):.1f}MB"
            line = f"{model.primary_type:<12} {model.filename[:40]:<40} {model.confidence:<10.2f} {size_mb:<10}"
            
            attr = curses.color_pair(2) if i == self.selected_row else 0
            self.stdscr.addstr(y, 0, line[:self.width-1], attr)
        
        # Help line
        help_line = "Keys: ↑/↓ Select | ←/→ Page | ENTER/d Details | f Filter | s Search | r Reset | h Help | q Quit"
        self.stdscr.addstr(self.height-2, 0, help_line[:self.width-1], curses.color_pair(3))
        
        # Status line
        if self.status_message:
            self.stdscr.addstr(self.height-1, 0, self.status_message[:self.width-1], curses.color_pair(3))
        
        self.stdscr.refresh()
    
    def show_model_details(self):
        if not self.models or self.selected_row >= len(self.models):
            return
        
        model = self.models[self.selected_row]
        
        # Create detail window
        detail_win = curses.newwin(self.height-4, self.width-4, 2, 2)
        detail_win.box()
        detail_win.addstr(1, 2, f"Model Details - {model.filename}", curses.A_BOLD)
        
        y = 3
        details = [
            f"ID: {model.id}",
            f"Filename: {model.filename}",
            f"Type: {model.primary_type} / {model.sub_type}",
            f"Confidence: {model.confidence:.2f}",
            f"Size: {model.file_size / (1024*1024):.1f} MB",
            f"Architecture: {model.architecture or 'N/A'}",
            f"Precision: {model.precision or 'N/A'}",
            f"Triggers: {model.triggers or 'N/A'}",
            f"Classified: {model.classified_at}",
        ]
        
        for detail in details:
            if y >= self.height - 8:
                break
            detail_win.addstr(y, 2, detail[:self.width-8])
            y += 1
        
        # Show metadata
        try:
            metadata = self.db.get_model_metadata(model.id)
            if metadata:
                y += 1
                detail_win.addstr(y, 2, "Metadata:", curses.A_BOLD)
                y += 1
                for key, value in metadata:
                    if y >= self.height - 8:
                        break
                    detail_win.addstr(y, 4, f"{key}: {value}"[:self.width-10])
                    y += 1
        except Exception as e:
            detail_win.addstr(y, 2, f"Error loading metadata: {e}")
        
        detail_win.addstr(self.height-7, 2, "Press any key to continue...")
        detail_win.refresh()
        detail_win.getch()
        del detail_win
    
    def filter_models(self):
        # Show available types
        filter_win = curses.newwin(min(len(self.model_types) + 6, self.height-4), 40, 2, 2)
        filter_win.box()
        filter_win.addstr(1, 2, "Select Model Type Filter:", curses.A_BOLD)
        
        options = ["(Clear filter)"] + [f"{t[0]} ({t[1]})" for t in self.model_types]
        selected = 0
        
        while True:
            for i, option in enumerate(options):
                y = 3 + i
                if y >= filter_win.getmaxyx()[0] - 2:
                    break
                attr = curses.color_pair(2) if i == selected else 0
                filter_win.addstr(y, 2, option[:36], attr)
            
            filter_win.addstr(filter_win.getmaxyx()[0]-2, 2, "↑/↓ Select, ENTER Confirm, ESC Cancel")
            filter_win.refresh()
            
            key = filter_win.getch()
            if key == 27:  # ESC
                break
            elif key == curses.KEY_UP and selected > 0:
                selected -= 1
            elif key == curses.KEY_DOWN and selected < len(options) - 1:
                selected += 1
            elif key == ord('\n') or key == curses.KEY_ENTER:
                if selected == 0:
                    self.current_filter = None
                else:
                    self.current_filter = self.model_types[selected-1][0]
                self.current_page = 0
                self.selected_row = 0
                self.load_models()
                break
        
        del filter_win
    
    def search_models(self):
        curses.curs_set(1)  # Show cursor
        curses.echo()
        
        search_win = curses.newwin(5, 60, self.height//2-2, self.width//2-30)
        search_win.box()
        search_win.addstr(1, 2, "Search in filenames and triggers:")
        search_win.addstr(2, 2, "> ")
        search_win.refresh()
        
        try:
            search_term = search_win.getstr(2, 4, 50).decode('utf-8').strip()
            if search_term:
                self.search_term = search_term
                self.current_page = 0
                self.selected_row = 0
                self.load_models()
        except:
            pass
        
        curses.noecho()
        curses.curs_set(0)
        del search_win
    
    def reset_filters(self):
        self.current_filter = None
        self.search_term = None
        self.current_page = 0
        self.selected_row = 0
        self.load_models()
        self.status_message = "Filters reset"
    
    def show_help(self):
        help_text = [
            "ModelHub CLI Help",
            "",
            "Navigation:",
            "  ↑/↓ - Move selection up/down",
            "  ←/→ - Previous/next page",
            "  ENTER or 'd' - Show model details",
            "",
            "Filtering & Search:",
            "  'f' - Filter by model type",
            "  's' - Search in filenames/triggers",
            "  'r' - Reset all filters",
            "",
            "Other:",
            "  'h' - Show this help",
            "  'q' - Quit application",
            "",
            "Press any key to continue..."
        ]
        
        help_win = curses.newwin(len(help_text) + 4, 50, 2, 2)
        help_win.box()
        
        for i, line in enumerate(help_text):
            help_win.addstr(i + 2, 2, line)
        
        help_win.refresh()
        help_win.getch()
        del help_win


def main():
    db_path = os.path.join(os.path.dirname(__file__), 'modelhub.db')
    
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        sys.exit(1)
    
    try:
        cli = ModelHubCLI(db_path)
        curses.wrapper(cli.run)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if 'cli' in locals():
            cli.db.close()


if __name__ == "__main__":
    main()