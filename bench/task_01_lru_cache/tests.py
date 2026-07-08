import pytest
import os
import sys

# Ensure we can import solution.py from the same directory
sys.path.insert(0, os.path.dirname(__file__))

def test_lru_cache_basic():
    from solution import LRUCache
    
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1
    
    cache.put(3, 3) # evicts key 2
    assert cache.get(2) == -1
    
    cache.put(4, 4) # evicts key 1
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4

def test_lru_cache_update():
    from solution import LRUCache
    
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(1, 10) # update key 1
    
    cache.put(3, 3) # evicts key 2 (key 1 was recently used!)
    assert cache.get(2) == -1
    assert cache.get(1) == 10

def test_lru_cache_capacity_one():
    from solution import LRUCache
    
    cache = LRUCache(1)
    cache.put(1, 1)
    assert cache.get(1) == 1
    cache.put(2, 2)
    assert cache.get(1) == -1
    assert cache.get(2) == 2
