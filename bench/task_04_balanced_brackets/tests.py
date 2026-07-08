import pytest
import os
import sys

# Ensure we can import solution.py from the same directory
sys.path.insert(0, os.path.dirname(__file__))

def test_balanced_brackets_basic():
    from solution import isValid
    
    assert isValid("()") is True
    assert isValid("()[]{}") is True
    assert isValid("(]") is False
    assert isValid("([)]") is False
    assert isValid("{[]}") is True

def test_balanced_brackets_edge():
    from solution import isValid
    
    assert isValid("") is True
    assert isValid("[") is False
    assert isValid("]") is False
    assert isValid("]}") is False
