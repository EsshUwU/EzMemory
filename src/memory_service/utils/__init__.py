"""
Utilities module initialization.
"""

from .hashing import generate_content_hash, generate_short_hash
from .time import (
    get_current_timestamp,
    timestamp_to_string,
    string_to_timestamp,
    is_expired,
    days_since,
)
from .text import normalize_text, truncate_text, extract_keywords
from .add_to_agents import add_cursor_mcp, add_vscode_mcp

__all__ = [
    "generate_content_hash",
    "generate_short_hash",
    "get_current_timestamp",
    "timestamp_to_string",
    "string_to_timestamp",
    "is_expired",
    "days_since",
    "normalize_text",
    "truncate_text",
    "extract_keywords",
    "add_cursor_mcp",
    "add_vscode_mcp",
]
