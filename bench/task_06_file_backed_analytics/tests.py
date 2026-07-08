import os
import sys
import json

import pytest

sys.path.insert(0, os.path.dirname(__file__))


def test_course_summary_uses_bundled_json_and_csv():
    from solution import LearningAnalytics

    analytics = LearningAnalytics()

    py_summary = analytics.course_summary("py101")
    assert py_summary == {
        "course_id": "py101",
        "title": "Python Foundations",
        "capacity": 4,
        "enrolled": 4,
        "completed": 2,
        "completion_rate": 0.5,
        "average_score": 88.0,
        "is_over_capacity": False,
    }

    ml_summary = analytics.course_summary("ml301")
    assert ml_summary["title"] == "Applied Machine Learning"
    assert ml_summary["enrolled"] == 2
    assert ml_summary["completed"] == 1
    assert ml_summary["completion_rate"] == 0.5
    assert ml_summary["average_score"] == 91.0
    assert ml_summary["is_over_capacity"] is False


def test_learner_report_preserves_file_order_and_completed_score_average():
    from solution import LearningAnalytics

    report = LearningAnalytics().learner_report("u1")

    assert report == {
        "learner_id": "u1",
        "courses": ["py101", "sql201"],
        "completed_courses": 2,
        "total_minutes": 310,
        "average_completed_score": 90.0,
    }


def test_unknown_course_and_empty_learner_report():
    from solution import LearningAnalytics

    analytics = LearningAnalytics()

    with pytest.raises(KeyError):
        analytics.course_summary("missing")

    assert analytics.learner_report("nobody") == {
        "learner_id": "nobody",
        "courses": [],
        "completed_courses": 0,
        "total_minutes": 0,
        "average_completed_score": 0.0,
    }


def test_explicit_data_dir_is_supported():
    from solution import LearningAnalytics

    data_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    analytics = LearningAnalytics(data_dir=data_dir)

    sql_summary = analytics.course_summary("sql201")
    assert sql_summary["enrolled"] == 3
    assert sql_summary["completed"] == 2
    assert sql_summary["completion_rate"] == 0.667
    assert sql_summary["average_score"] == 83.5


def test_course_summary_edge_cases_from_temp_fixtures(tmp_path):
    from solution import LearningAnalytics

    courses = [
        {
            "id": "packed",
            "title": "Packed Course",
            "capacity": 1,
            "tags": [],
        },
        {
            "id": "empty",
            "title": "No Enrollment Course",
            "capacity": 5,
            "tags": [],
        },
    ]
    (tmp_path / "courses.json").write_text(json.dumps(courses), encoding="utf-8")
    (tmp_path / "enrollments.csv").write_text(
        "\n".join([
            "learner_id,course_id,status,score,minutes",
            "u1,packed,in_progress,73,40",
            "u2,packed,dropped,0,12",
        ]),
        encoding="utf-8",
    )

    analytics = LearningAnalytics(data_dir=str(tmp_path))

    packed = analytics.course_summary("packed")
    assert packed["enrolled"] == 2
    assert packed["completed"] == 0
    assert packed["completion_rate"] == 0.0
    assert packed["average_score"] == 0.0
    assert packed["is_over_capacity"] is True

    empty = analytics.course_summary("empty")
    assert empty["enrolled"] == 0
    assert empty["completed"] == 0
    assert empty["completion_rate"] == 0.0
    assert empty["average_score"] == 0.0
    assert empty["is_over_capacity"] is False
