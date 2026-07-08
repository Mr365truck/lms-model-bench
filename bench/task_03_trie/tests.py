import pytest
import os
import sys

# Ensure we can import solution.py from the same directory
sys.path.insert(0, os.path.dirname(__file__))

def test_trie_basic():
    from solution import Trie
    
    trie = Trie()
    trie.insert("apple")
    assert trie.search("apple") is True
    assert trie.search("app") is False
    assert trie.startsWith("app") is True
    trie.insert("app")
    assert trie.search("app") is True

def test_trie_empty_and_overlap():
    from solution import Trie
    
    trie = Trie()
    assert trie.search("") is False
    assert trie.startsWith("") is True # Empty prefix is prefix of everything
    trie.insert("a")
    assert trie.search("a") is True
    assert trie.startsWith("a") is True
    assert trie.search("ab") is False
