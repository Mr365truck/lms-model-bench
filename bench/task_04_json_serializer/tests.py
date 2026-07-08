import pytest
import os
import sys
import datetime
import json

sys.path.insert(0, os.path.dirname(__file__))

def test_serialize_basic():
    from solution import serialize_custom
    
    data = {
        "name": "Ethan",
        "age": 30,
        "is_developer": True,
        "scores": [95, 88.5, None],
        "created_at": datetime.datetime(2026, 7, 8, 14, 0, 0)
    }
    
    serialized = serialize_custom(data)
    parsed = json.loads(serialized)
    
    assert parsed["name"] == "Ethan"
    assert parsed["age"] == 30
    assert parsed["is_developer"] is True
    assert parsed["scores"] == [95, 88.5, None]
    assert parsed["created_at"] == "2026-07-08T14:00:00"

def test_serialize_custom_objects():
    from solution import serialize_custom
    
    class Model:
        def __init__(self, val):
            self.val = val
        def to_json(self):
            return {"value": self.val}

    class SimpleObj:
        def __init__(self):
            self.x = 10
            self.y = "hello"

    m = Model(SimpleObj())
    serialized = serialize_custom(m)
    parsed = json.loads(serialized)
    assert parsed == {"value": {"x": 10, "y": "hello"}}

def test_serialize_circular_references():
    from solution import serialize_custom
    
    # Direct cycle
    lst = []
    lst.append(lst)
    with pytest.raises(ValueError) as exc:
        serialize_custom(lst)
    assert "circular" in str(exc.value).lower()

    # Transitive cycle
    a = {}
    b = {"a_ref": a}
    a["b_ref"] = b
    with pytest.raises(ValueError) as exc:
        serialize_custom(a)
    assert "circular" in str(exc.value).lower()
