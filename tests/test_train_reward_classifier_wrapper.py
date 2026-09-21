from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "train_reward_classifier.sh"


def test_wrapper_uses_repo_local_trainer_not_missing_upstream_copy():
    source = WRAPPER.read_text(encoding="utf-8")

    assert 'TRAIN_SCRIPT="${SERL_FR3_ROOT}/scripts/train_reward_classifier.py"' in source
    assert "upstream/hil-serl/serl_launcher/utils/train_reward_classifier.py" not in source
    assert "serl_launcher.utils.train_reward_classifier" not in source


def test_wrapper_passes_flags_supported_by_local_trainer():
    source = WRAPPER.read_text(encoding="utf-8")
    cmd_section = source.split("CMD_ARGS=(", 1)[1].split(")", 1)[0]

    assert "--data-dir" in cmd_section
    assert "--output-dir" in cmd_section
    assert "--batch-size" in cmd_section
    assert "--data_dir" not in cmd_section
    assert "--output_dir" not in cmd_section
    assert "--batch_size" not in cmd_section
    assert "--image_keys" not in cmd_section
