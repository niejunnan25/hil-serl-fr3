"""teach_session pure-helper tests (labeling + index record + label parsing)."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from teach_session import labeled_path, make_index_record, parse_label  # noqa: E402


class TestLabeledPath:
    def test_inserts_label_before_ext(self):
        assert labeled_path("/d/gello_demo_20260615_1.pkl", "success") == \
            "/d/gello_demo_20260615_1_success.pkl"
        assert labeled_path("/d/gello_demo_1_raw.npz", "fail") == \
            "/d/gello_demo_1_raw_fail.npz"


class TestParseLabel:
    def test_parses_variants(self):
        assert parse_label("s") == "success"
        assert parse_label("success") == "success"
        assert parse_label("f") == "fail"
        assert parse_label("d") == "discard"
        assert parse_label("DISCARD") == "discard"

    def test_default_is_fail_conservative(self):
        assert parse_label("") == "fail"
        assert parse_label("xyz") == "fail"


class TestMakeIndexRecord:
    def _res(self):
        return {
            "pkl": "/d/gello_demo_20260615_133341.pkl",
            "npz": "/d/gello_demo_20260615_133341_raw.npz",
            "n_transitions": 478,
            "elapsed_s": 48.3,
            "validate_ok": True,
            "abort_reason": "signal 15",
        }

    def test_record_uses_labeled_basenames_and_fields(self):
        rec = make_index_record(self._res(), "success", "clean grasp", "gello_demo_20260615_133341")
        assert rec["label"] == "success"
        assert rec["notes"] == "clean grasp"
        assert rec["pkl"] == "gello_demo_20260615_133341_success.pkl"
        assert rec["npz"] == "gello_demo_20260615_133341_raw_success.npz"
        assert rec["n_transitions"] == 478
        assert rec["elapsed_s"] == 48.3
        assert rec["validate_ok"] is True
        assert rec["abort_reason"] == "signal 15"

    def test_record_is_json_serializable(self):
        import json

        rec = make_index_record(self._res(), "fail", "missed socket", "ts1")
        s = json.dumps(rec, ensure_ascii=False)
        assert json.loads(s)["label"] == "fail"
