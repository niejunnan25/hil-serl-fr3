import ast
from dataclasses import replace
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from hilserl.config import Config
from hilserl.image_profile import (FRONT_ROI160, FRONT_SIDE_CROP, FULL_FRAME, INSERT_ROI,
                                    PROFILE_NAMES, get_image_profile, preprocess_image)
from hilserl.learning_replay import validate_transition
from test_fixed_xyz_learning_integration import transition

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("camera,box", [("wrist_1", (360,0,720,720)), ("side_policy",(180,280,400,400))])
def test_policy_crop_uses_original_pixels_and_preserves_rgb(camera, box):
    rng = np.random.default_rng(0)
    frame = rng.integers(0,256,size=(720,1280,3),dtype=np.uint8)
    x,y,w,h=box
    expected = cv2.resize(frame[y:y+h,x:x+w],(224,224),interpolation=cv2.INTER_AREA)[...,::-1]
    actual = preprocess_image(frame,camera,INSERT_ROI)
    np.testing.assert_array_equal(actual,expected)
    assert actual.shape == (224,224,3)
    assert actual.flags.c_contiguous


def test_classifier_pixels_identical_to_legacy_and_crop_rejects_wrong_camera_size():
    frame=np.random.default_rng(2).integers(0,256,(720,1280,3),np.uint8)
    np.testing.assert_array_equal(preprocess_image(frame,"side_classifier",INSERT_ROI),
                                  preprocess_image(frame,"side_classifier",FULL_FRAME))
    with pytest.raises(ValueError,match="dimensions"):
        preprocess_image(np.zeros((128,128,3),np.uint8),"wrist_1",INSERT_ROI)
    changed=get_image_profile(INSERT_ROI);changed["cameras"]["wrist_1"]["crop_xywh"][0]+=1
    with pytest.raises(ValueError,match="geometry"):
        preprocess_image(frame,"wrist_1",changed)


@pytest.mark.parametrize("size", [128,160,192,224])
def test_resolution_profiles_share_roi_but_bind_output_and_classifier(size):
    profile=get_image_profile(f"insert-roi{size}-v1")
    frame=np.zeros((720,1280,3),np.uint8)
    for camera in ("wrist_1","side_policy"):
        assert profile["cameras"][camera]["crop_xywh"]==get_image_profile(INSERT_ROI)["cameras"][camera]["crop_xywh"]
        assert preprocess_image(frame,camera,profile).shape==(size,size,3)
    assert preprocess_image(frame,"side_classifier",profile).shape==(128,128,3)


def test_production_camera_read_reuses_side_frame_for_policy_and_classifier():
    tree=ast.parse((ROOT/"experiments/plug_insertion/env.py").read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="PlugInsertionEnv")
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="get_im")
    namespace={}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),"actual_get_im","exec"),namespace)
    frame=np.random.default_rng(3).integers(0,256,(720,1280,3),np.uint8)
    class Capture:
        last_read={"camera":"side_policy","capture_id":42,"monotonic_ns":100}
        count=0
        def read(self):self.count+=1;return frame
    wrist,side=Capture(),Capture()
    env=SimpleNamespace(cap={"wrist_1":wrist,"side_policy":side,"side_classifier":side},
          config=SimpleNamespace(IMAGE_PROFILE=INSERT_ROI),frame_references={},display_image=False,
          observation_space={"images":{k:SimpleNamespace(shape=(s,s,3)) for k,s in
                             (("wrist_1",224),("side_policy",224),("side_classifier",128))}})
    images=namespace["get_im"](env)
    assert side.count==wrist.count==1
    assert env.frame_references["side_policy"]==env.frame_references["side_classifier"]
    for key in images:np.testing.assert_array_equal(images[key],preprocess_image(frame,key,INSERT_ROI))


def test_replay_rejects_fullframe_data_or_wrong_crop_version():
    data=transition()
    with pytest.raises(ValueError,match="profile"):
        validate_transition(data,INSERT_ROI)
    data["infos"]["image_profile"]=INSERT_ROI
    with pytest.raises(ValueError,match="observations"):
        validate_transition(data,INSERT_ROI)
    for field in ("observations","next_observations"):
        for key in ("wrist_1","side_policy"):data[field][key]=np.zeros((1,224,224,3),np.uint8)
    assert validate_transition(data,INSERT_ROI).endswith("/000000")
    with pytest.raises(ValueError,match="profile"):validate_transition(data,FULL_FRAME)


def test_config_missing_image_version_stays_128_and_hidden_env_cannot_override(monkeypatch,tmp_path):
    monkeypatch.setenv("HILSERL_IMAGE_PROFILE",INSERT_ROI)
    legacy=replace(Config(),root=tmp_path)
    assert legacy.environment("actor",run_dir=tmp_path)["HILSERL_IMAGE_PROFILE"]==FULL_FRAME
    current=replace(legacy,action_contract="fixed-xyz-v1",image_profile=INSERT_ROI)
    assert current.validate().environment("actor",run_dir=tmp_path)["HILSERL_IMAGE_PROFILE"]==INSERT_ROI
    with pytest.raises(ValueError):replace(legacy,image_profile=INSERT_ROI).validate()


def test_new_recording_seed_builder_and_loader_keep_the_image_version(tmp_path):
    import json
    from hilserl import seed_dataset as seed
    from hilserl.storage import read_step, write_step
    from test_seed_dataset import make_episode
    run,directory=make_episode(tmp_path)
    config_path=run/"attempts/config.json"
    config=json.loads(config_path.read_text());config["image_profile"]=INSERT_ROI
    config_path.write_text(json.dumps(config))
    for path in directory.glob("*.npz"):
        raw=read_step(path);raw["image_profile"]=INSERT_ROI
        for field in ("observations","next_observations"):
            for key in ("wrist_1","side_policy"):
                # Synthetic camera frames for schema propagation, not a recrop
                # of historical low-resolution images.
                value=raw[field][key][0,0,0,0]
                raw[field][key]=np.full((1,224,224,3),value,np.uint8)
        write_step(path,raw)
    output=tmp_path/"cropped-recording-seed"
    manifest=seed.build_seed_dataset([run],output)
    assert manifest["image_profile"]==get_image_profile(INSERT_ROI)
    with pytest.raises(ValueError,match="image profile"):seed.validate_seed_dataset(output)
    transitions=list(seed.iter_seed_transitions(output,expected_image_profile=INSERT_ROI))
    assert len(transitions)==3 and sum(t["rewards"] for t in transitions)==1
    for t in transitions:
        assert t["infos"]["image_profile"]==INSERT_ROI
        assert t["observations"]["side_policy"].shape==(1,224,224,3)
        assert t["observations"]["side_classifier"].shape==(1,128,128,3)


def test_export_preserves_roi_profile_and_rejects_mixed_frame_contract(tmp_path):
    import json
    from hilserl.storage import read_step,write_step
    from test_fixed_xyz_export import recording,exported
    archive,path=recording(tmp_path)
    for file in (path/"manifest.json",path/"episodes/000001/episode.json"):
        meta=json.loads(file.read_text())
        if file.name=="manifest.json":meta["metadata"]["image_profile"]=INSERT_ROI
        else:meta["image_profile"]=INSERT_ROI
        file.write_text(json.dumps(meta))
    for file in (path/"episodes/000001").glob("*.npz"):
        raw=read_step(file);raw["image_profile"]=INSERT_ROI;write_step(file,raw)
    manifest,transitions=exported(archive,format="serl")
    assert manifest["image_profiles"]==[INSERT_ROI]
    assert all(t["infos"]["image_profile"]==INSERT_ROI for t in transitions)
    file=path/"episodes/000001/000002.npz"
    raw=read_step(file);raw["image_profile"]=FULL_FRAME;write_step(file,raw)
    with pytest.raises(ValueError,match="profiles differ"):archive.export("run/capture",source="human",format="serl")


@pytest.mark.parametrize("size", [128, 160, 192, 224])
def test_front_mount_family_moves_only_the_external_camera(size):
    profile = get_image_profile(f"insert-front-roi{size}-v1")
    side = get_image_profile("insert-roi160-v1")
    assert profile["cameras"]["wrist_1"]["crop_xywh"] == side["cameras"]["wrist_1"]["crop_xywh"]
    assert profile["cameras"]["side_policy"]["crop_xywh"] == list(FRONT_SIDE_CROP)
    assert profile["cameras"]["side_classifier"]["crop_xywh"] is None
    assert profile["cameras"]["side_classifier"]["interpolation"] == "area"
    frame = np.zeros((720, 1280, 3), np.uint8)
    for camera in ("wrist_1", "side_policy"):
        assert preprocess_image(frame, camera, profile).shape == (size, size, 3)
    assert preprocess_image(frame, "side_classifier", profile).shape == (128, 128, 3)


def test_front_crop_reads_original_pixels():
    rng = np.random.default_rng(11)
    frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    x, y, w, h = FRONT_SIDE_CROP
    expected = cv2.resize(frame[y:y+h, x:x+w], (160, 160), interpolation=cv2.INTER_AREA)[..., ::-1]
    np.testing.assert_array_equal(preprocess_image(frame, "side_policy", FRONT_ROI160), expected)


def test_every_declared_profile_name_has_geometry(tmp_path):
    for name in PROFILE_NAMES:
        profile = get_image_profile(name)
        assert profile["name"] == name
    front = get_image_profile(FRONT_ROI160)
    assert front != get_image_profile("insert-roi160-v1")
    cfg = replace(Config(), root=tmp_path, action_contract="fixed-xyz-v1", image_profile=FRONT_ROI160)
    assert cfg.validate().image_profile == FRONT_ROI160
    with pytest.raises(ValueError, match="Unknown image_profile"):
        replace(Config(), root=tmp_path, action_contract="fixed-xyz-v1",
                image_profile="insert-front-roi160-v99").validate()


def test_cli_config_import_needs_no_camera_or_math_stack():
    """`bin/hil-serl` runs under a bare interpreter; profile validation must not
    pull in numpy/cv2 through hilserl.config."""
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'cv2', 'numpy', 'torch', 'jax'}:\n"
        "            raise ImportError('blocked ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import hilserl.config, hilserl.profile_names\n"
        "print(hilserl.config.load_config().image_profile)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
