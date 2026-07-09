import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def test_normalize_slug():
    from solution import normalize_slug

    assert normalize_slug("  Hello, LMS World!  ") == "hello-lms-world"
    assert normalize_slug("Python___Testing---Basics") == "python-testing-basics"
    assert normalize_slug("###") == ""
    assert normalize_slug("Version 2.0: Ready?") == "version-2-0-ready"


def test_summarize_words():
    from solution import summarize_words

    assert summarize_words("Cat dog cat! DOG bird.") == {
        "word_count": 5,
        "unique_count": 3,
        "top_word": "cat",
    }
    assert summarize_words("beta alpha beta alpha") == {
        "word_count": 4,
        "unique_count": 2,
        "top_word": "alpha",
    }
    assert summarize_words("1234 !!!") == {
        "word_count": 0,
        "unique_count": 0,
        "top_word": None,
    }
