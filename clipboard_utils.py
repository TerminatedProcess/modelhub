#!/usr/bin/env python3
"""
Clipboard utility functions for ModelHub
Cross-platform clipboard support for Linux (X11/Wayland), macOS, and Windows
"""

import subprocess
import shutil
import sys
from typing import Optional

# Cache the detected clipboard tool to avoid repeated detection
_cached_clipboard_tool = None

def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to system clipboard (optimized version with caching).
    
    Args:
        text: The text to copy to clipboard
        
    Returns:
        bool: True if successful, False if failed
    """
    global _cached_clipboard_tool
    
    if not text:
        return False
    
    # Detect clipboard tool once and cache it
    if _cached_clipboard_tool is None:
        if shutil.which("wl-copy"):
            _cached_clipboard_tool = "wl-copy"
        elif shutil.which("xclip"):
            _cached_clipboard_tool = "xclip"
        elif shutil.which("xsel"):
            _cached_clipboard_tool = "xsel"
        elif shutil.which("pbcopy"):
            _cached_clipboard_tool = "pbcopy"
        elif shutil.which("clip"):
            _cached_clipboard_tool = "clip"
        else:
            try:
                import pyperclip
                _cached_clipboard_tool = "pyperclip"
            except ImportError:
                _cached_clipboard_tool = "none"
    
    # Use cached tool for fast execution
    try:
        if _cached_clipboard_tool == "wl-copy":
            subprocess.run(["wl-copy"], input=text, text=True, check=False)
            return True
            
        elif _cached_clipboard_tool == "xclip":
            subprocess.run(["xclip", "-selection", "clipboard"], input=text, text=True, check=False)
            return True
            
        elif _cached_clipboard_tool == "xsel":
            subprocess.run(["xsel", "--clipboard", "--input"], input=text, text=True, check=False)
            return True
            
        elif _cached_clipboard_tool == "pbcopy":
            result = subprocess.run(
                ["pbcopy"], 
                input=text, 
                text=True, 
                capture_output=True,
                timeout=1  # Reduced from 5 to 1 second
            )
            return result.returncode == 0
            
        elif _cached_clipboard_tool == "clip":
            result = subprocess.run(
                ["clip"], 
                input=text, 
                text=True, 
                capture_output=True,
                timeout=1  # Reduced from 5 to 1 second
            )
            return result.returncode == 0
            
        elif _cached_clipboard_tool == "pyperclip":
            import pyperclip
            pyperclip.copy(text)
            return True
            
        else:
            return False
                
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, Exception):
        return False

def get_clipboard_tools_status() -> dict:
    """
    Check which clipboard tools are available on the system.
    
    Returns:
        dict: Status of available clipboard tools
    """
    tools = {
        "wl-copy": shutil.which("wl-copy") is not None,
        "xclip": shutil.which("xclip") is not None, 
        "xsel": shutil.which("xsel") is not None,
        "pbcopy": shutil.which("pbcopy") is not None,
        "clip": shutil.which("clip") is not None,
        "pyperclip": False
    }
    
    try:
        import pyperclip
        tools["pyperclip"] = True
    except ImportError:
        pass
        
    return tools

def get_preferred_clipboard_tool() -> Optional[str]:
    """
    Get the preferred clipboard tool for the current system.
    
    Returns:
        str: Name of the preferred tool, or None if none available
    """
    if shutil.which("wl-copy"):
        return "wl-copy (Wayland)"
    elif shutil.which("xclip"):
        return "xclip (X11)"
    elif shutil.which("xsel"):
        return "xsel (X11)"
    elif shutil.which("pbcopy"):
        return "pbcopy (macOS)"
    elif shutil.which("clip"):
        return "clip (Windows)"
    else:
        try:
            import pyperclip
            return "pyperclip (Python library)"
        except ImportError:
            return None

if __name__ == "__main__":
    # Test the clipboard functionality
    test_text = "ModelHub clipboard test - this text should be in your clipboard!"
    
    print("Testing clipboard functionality...")
    print(f"Platform: {sys.platform}")
    print(f"Available tools: {get_clipboard_tools_status()}")
    print(f"Preferred tool: {get_preferred_clipboard_tool()}")
    
    if copy_to_clipboard(test_text):
        print(f"✓ Successfully copied to clipboard: '{test_text}'")
    else:
        print("✗ Failed to copy to clipboard")
        print("\nTo fix this, install one of:")
        print("  - Wayland: wl-clipboard (wl-copy)")
        print("  - X11: xclip or xsel")
        print("  - Python: pip install pyperclip")