import pytest
import os
import sys

# Ensure we can import solution.py from the same directory
sys.path.insert(0, os.path.dirname(__file__))

def test_binary_search_basic():
    from solution import searchRange
    
    assert searchRange([5,7,7,8,8,10], 8) == [3, 4]
    assert searchRange([5,7,7,8,8,10], 6) == [-1, -1]
    assert searchRange([], 0) == [-1, -1]

def test_binary_search_single():
    from solution import searchRange
    
    assert searchRange([1], 1) == [0, 0]
    assert searchRange([1], 2) == [-1, -1]

def test_binary_search_all_same():
    from solution import searchRange
    
    assert searchRange([2,2,2,2,2], 2) == [0, 4]
    assert searchRange([2,2,2,2,2], 3) == [-1, -1]
