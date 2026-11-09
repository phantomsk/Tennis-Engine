from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_starts_without_errors_and_prompts_for_upload():
    at = AppTest.from_file(APP, default_timeout=30).run()

    assert not at.exception
    assert at.title[0].value == "Tennis Strategy Engine"
