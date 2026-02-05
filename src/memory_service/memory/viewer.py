"""
Memory viewer - formatted display of memories.
"""

from typing import List
from datetime import datetime
from .schemas import Memory


def format_memory_date(dt: datetime) -> str:
    """
    Format datetime for display showing day and time.
    
    Args:
        dt: Datetime to format
        
    Returns:
        Formatted string like "Monday, Feb 3, 2026 at 14:30:45"
    """
    return dt.strftime("%A, %b %d, %Y at %H:%M:%S")


def format_memory_for_display(memory: Memory, index: int = None) -> str:
    """
    Format a single memory for display.
    
    Args:
        memory: Memory to format
        index: Optional index number
        
    Returns:
        Formatted string
    """
    lines = []
    
    # Header with optional index
    if index is not None:
        lines.append(f"[bold cyan]Memory #{index + 1}[/bold cyan]")
    else:
        lines.append("[bold cyan]Memory[/bold cyan]")
    
    # ID (shortened)
    if memory.id:
        short_id = memory.id[:12] + "..." if len(memory.id) > 12 else memory.id
        lines.append(f"[dim]ID:[/dim] {short_id}")
    
    # Content
    lines.append(f"[bold]Content:[/bold] {memory.content}")
    
    # Created date/time
    created_str = format_memory_date(memory.created_at)
    lines.append(f"[dim]Created:[/dim] {created_str}")
    
    # Accessed date/time (if available)
    if memory.accessed_at:
        accessed_str = format_memory_date(memory.accessed_at)
        lines.append(f"[dim]Last Accessed:[/dim] {accessed_str}")
    
    # Access count
    if memory.access_count > 0:
        lines.append(f"[dim]Access Count:[/dim] {memory.access_count}")
    
    # Tags (if any)
    if memory.metadata.tags:
        tags_str = ", ".join(memory.metadata.tags)
        lines.append(f"[dim]Tags:[/dim] {tags_str}")
    
    # Source (if any)
    if memory.metadata.source:
        lines.append(f"[dim]Source:[/dim] {memory.metadata.source}")
    
    return "\n".join(lines)


def format_memories_table(memories: List[Memory]) -> List[List[str]]:
    """
    Format memories as table rows for Rich Table.
    
    Args:
        memories: List of memories to format
        
    Returns:
        List of rows, each row is [index, content_preview, created_at, tags]
    """
    rows = []
    for idx, memory in enumerate(memories):
        # Truncate content for table display
        content_preview = memory.content
        if len(content_preview) > 60:
            content_preview = content_preview[:57] + "..."
        
        # Format date
        date_str = memory.created_at.strftime("%Y-%m-%d %H:%M")
        
        # Format tags
        tags_str = ", ".join(memory.metadata.tags[:3])  # Show first 3 tags
        if len(memory.metadata.tags) > 3:
            tags_str += "..."
        if not tags_str:
            tags_str = "[dim]none[/dim]"
        
        rows.append([
            str(idx + 1),
            content_preview,
            date_str,
            tags_str,
        ])
    
    return rows
