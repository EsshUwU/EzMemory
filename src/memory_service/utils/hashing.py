import hashlib
import uuid
from typing import Union


def generate_content_hash(content: str) -> str:
    """
    Generate a hash for content deduplication.
    
    Args:
        content: Text content to hash
        
    Returns:
        UUID string formatted from content hash
    """
    # Generate SHA256 hash
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    # Convert first 32 hex characters (128 bits) to UUID format
    # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    uuid_string = f"{content_hash[:8]}-{content_hash[8:12]}-{content_hash[12:16]}-{content_hash[16:20]}-{content_hash[20:32]}"
    
    return uuid_string


def generate_short_hash(content: str, length: int = 8) -> str:
    """
    Generate a short hash for display purposes.
    
    Args:
        content: Text content to hash
        length: Length of the hash
        
    Returns:
        Short hash string
    """
    full_hash = generate_content_hash(content)
    return full_hash[:length]
