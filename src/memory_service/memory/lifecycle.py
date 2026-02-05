"""
Memory lifecycle management - aging, decay, and pruning.
"""

from typing import List
from .schemas import Memory
from .storage import MemoryStorage
from ..utils.time import is_expired, days_since


class MemoryLifecycle:
    """Manages memory lifecycle operations."""
    
    def __init__(self, storage: MemoryStorage, ttl_days: int = 90):
        """
        Initialize lifecycle manager.
        
        Args:
            storage: Memory storage instance
            ttl_days: Time to live in days
        """
        self.storage = storage
        self.ttl_days = ttl_days
    
    def prune_expired(self) -> int:
        """
        Remove expired memories.
        
        Returns:
            Number of memories deleted
        """
        all_memories = self.storage.get_all()
        expired_ids = []
        
        for memory in all_memories:
            if is_expired(memory.created_at, self.ttl_days):
                expired_ids.append(memory.id)
        
        if expired_ids:
            self.storage.delete(expired_ids)
        
        return len(expired_ids)
    
    def calculate_decay_score(self, memory: Memory, decay_factor: float = 0.95) -> float:
        """
        Calculate decay score based on memory age.
        
        Args:
            memory: Memory to calculate score for
            decay_factor: Decay factor (0-1)
            
        Returns:
            Decay score
        """
        age_days = days_since(memory.created_at)
        decay_score = decay_factor ** age_days
        return decay_score
    
    def get_memory_health(self) -> dict:
        """
        Get overall memory health statistics.
        
        Returns:
            Dictionary with health metrics
        """
        all_memories = self.storage.get_all()
        total = len(all_memories)
        
        if total == 0:
            return {
                "total": 0,
                "expired": 0,
                "active": 0,
            }
        
        expired = sum(
            1 for m in all_memories
            if is_expired(m.created_at, self.ttl_days)
        )
        
        return {
            "total": total,
            "expired": expired,
            "active": total - expired,
        }
