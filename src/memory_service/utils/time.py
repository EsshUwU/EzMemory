from datetime import datetime, timedelta
from typing import Optional


def get_current_timestamp() -> datetime:
    """Get current UTC timestamp."""
    return datetime.utcnow()


def timestamp_to_string(timestamp: datetime) -> str:
    """Convert timestamp to ISO format string."""
    return timestamp.isoformat()


def string_to_timestamp(timestamp_str: str) -> datetime:
    """Convert ISO format string to timestamp."""
    return datetime.fromisoformat(timestamp_str)


def is_expired(
    created_at: datetime,
    ttl_days: int,
    current_time: Optional[datetime] = None
) -> bool:
    """
    Check if a memory has expired based on TTL.
    
    Args:
        created_at: Creation timestamp
        ttl_days: Time to live in days
        current_time: Current time (defaults to now)
        
    Returns:
        True if expired, False otherwise
    """
    if current_time is None:
        current_time = get_current_timestamp()
    
    expiry_time = created_at + timedelta(days=ttl_days)
    return current_time > expiry_time


def days_since(timestamp: datetime) -> int:
    """
    Calculate days since a timestamp.
    
    Args:
        timestamp: Past timestamp
        
    Returns:
        Number of days
    """
    delta = get_current_timestamp() - timestamp
    return delta.days
