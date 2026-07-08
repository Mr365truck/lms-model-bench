import pytest
import os
import sys

# Ensure we can import solution.py from the same directory
sys.path.insert(0, os.path.dirname(__file__))

def test_merge_intervals_basic():
    from solution import merge
    
    assert merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
    assert merge([[1,4],[4,5]]) == [[1,5]]

def test_merge_intervals_sorted():
    from solution import merge
    
    assert merge([[1,4],[0,4]]) == [[0,4]]
    assert merge([[1,4],[2,3]]) == [[1,4]]

def test_merge_intervals_empty_and_single():
    from solution import merge
    
    assert merge([]) == []
    assert merge([[1,4]]) == [[1,4]]
