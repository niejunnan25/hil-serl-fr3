import ast
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ZED_CAPTURE = REPO_ROOT / "scripts" / "zed_capture.py"
CALIBRATE_IMAGE_CROP = REPO_ROOT / "scripts" / "calibrate_image_crop.py"
PLUG_CONFIG = REPO_ROOT / "experiments" / "plug_insertion" / "config.py"


def _load_module(module_name: str, path: Path):
    sys.modules.pop(module_name, None)
    original_path = list(sys.path)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path


def _plug_camera_exposures() -> dict[str, int]:
    tree = ast.parse(PLUG_CONFIG.read_text(encoding="utf-8"), filename=str(PLUG_CONFIG))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "EnvConfig":
            for item in node.body:
                if not isinstance(item, ast.Assign):
                    continue
                if any(isinstance(t, ast.Name) and t.id == "REALSENSE_CAMERAS" for t in item.targets):
                    cameras = ast.literal_eval(item.value)
                    return {name: int(cfg["exposure"]) for name, cfg in cameras.items()}
    raise AssertionError("EnvConfig.REALSENSE_CAMERAS not found")


@pytest.mark.parametrize("value", [None, -1, 0, 32, 39, 100])
def test_zed_exposure_accepts_sdk_range(value):
    zed_capture = _load_module("_test_zed_capture_exposure", ZED_CAPTURE)

    assert zed_capture.validate_zed_exposure(value) == value


@pytest.mark.parametrize("value", [-2, 101, 10500, 13000])
def test_zed_exposure_rejects_realsense_microsecond_values(value):
    zed_capture = _load_module("_test_zed_capture_exposure", ZED_CAPTURE)

    with pytest.raises(ValueError, match="ZED exposure"):
        zed_capture.validate_zed_exposure(value)


def test_plug_insertion_zed_exposures_are_valid_sdk_percentages():
    zed_capture = _load_module("_test_zed_capture_exposure", ZED_CAPTURE)

    assert _plug_camera_exposures() == {
        "wrist_1": 32,
        "side_policy": 39,
        "side_classifier": 39,
    }
    for exposure in _plug_camera_exposures().values():
        assert zed_capture.validate_zed_exposure(exposure) == exposure


def test_calibration_default_exposure_matches_wrist_runtime_config():
    calibrate = _load_module("_test_calibrate_image_crop_exposure", CALIBRATE_IMAGE_CROP)

    assert calibrate.DEFAULT_ZED_EXPOSURE == _plug_camera_exposures()["wrist_1"]
    assert calibrate.parse_camera_arg("wrist_1:13132609").exposure == calibrate.DEFAULT_ZED_EXPOSURE
    assert calibrate.parse_camera_arg("side_policy:36276705:39").exposure == 39


def test_calibration_rejects_invalid_camera_arg_exposure():
    calibrate = _load_module("_test_calibrate_image_crop_exposure", CALIBRATE_IMAGE_CROP)

    with pytest.raises(ValueError, match="ZED exposure"):
        calibrate.parse_camera_arg("side_policy:36276705:13000")
