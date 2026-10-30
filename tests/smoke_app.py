"""Execute each Streamlit model path with training disabled by a guard."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from sklearn.pipeline import Pipeline
from src.core import ROOT, MODEL_FILES


def run():
    with patch.object(Pipeline, "fit", side_effect=AssertionError("App must never train")):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        assert not at.exception, str(at.exception)
        for name in MODEL_FILES:
            at.selectbox[0].select(name).run()
            assert not at.exception, str(at.exception)
            assert len(at.metric) == 6, f"Expected six metrics for {name}"
        at.radio[0].set_value("Upload test CSV").run()
        assert not at.exception, str(at.exception)
        assert any("Upload a labeled" in i.value for i in at.info)
    print("PASS: all 5 model views, 6 metrics each, upload waiting state; no fitting allowed.")


if __name__ == "__main__":
    run()
