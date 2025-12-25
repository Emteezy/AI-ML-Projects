"""
Redis caching utilities (optional)
"""
import json
from typing import Optional, Any
import pickle

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.config import REDIS_HOST, REDIS_PORT, REDIS_DB, USE_REDIS


class CacheManager:
    """Redis cache manager for recommendations"""
    
    def __init__(self):
        self.redis_client = None
        self.enabled = USE_REDIS and REDIS_AVAILABLE
        
        if self.enabled:
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    decode_responses=False
                )
                # Test connection
                self.redis_client.ping()
                print("Redis cache enabled")
            except Exception as e:
                print(f"Redis connection failed: {e}. Caching disabled.")
                self.enabled = False
        else:
            print("Redis cache disabled")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            print(f"Cache get error: {e}")
        
        return None
    
    def set(self, key: str, value: Any, expire: int = 3600):
        """Set value in cache"""
        if not self.enabled:
            return
        
        try:
            serialized = pickle.dumps(value)
            self.redis_client.setex(key, expire, serialized)
        except Exception as e:
            print(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete key from cache"""
        if not self.enabled:
            return
        
        try:
            self.redis_client.delete(key)
        except Exception as e:
            print(f"Cache delete error: {e}")
    
    def clear(self):
        """Clear all cache"""
        if not self.enabled:
            return
        
        try:
            self.redis_client.flushdb()
        except Exception as e:
            print(f"Cache clear error: {e}")


# Global cache instance
cache_manager = CacheManager()

