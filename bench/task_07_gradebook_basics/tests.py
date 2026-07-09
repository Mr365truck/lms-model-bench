import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))


def test_gradebook_average_and_letters():
    from solution import Gradebook

    book = Gradebook()
    book.add_score("Ada", 100)
    book.add_score("Ada", 80)
    book.add_score("Grace", 79.5)

    assert book.average("Ada") == 90.0
    assert book.letter_grade("Ada") == "A"
    assert book.average("Grace") == 79.5
    assert book.letter_grade("Grace") == "C"


def test_gradebook_validation_and_top_student():
    from solution import Gradebook

    book = Gradebook()
    with pytest.raises(ValueError):
        book.add_score("Bad", 101)

    with pytest.raises(KeyError):
        book.average("Missing")

    with pytest.raises(ValueError):
        book.top_student()

    book.add_score("Zoe", 90)
    book.add_score("Amy", 90)
    book.add_score("Max", 89)

    assert book.top_student() == "Amy"
