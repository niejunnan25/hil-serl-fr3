# Plug ZED Insertion

Owner: Codex primary, Human approval required before live robot use

This task folder is the isolated ZED-stereo variant of the FR3 HIL-SERL plug
insertion experiment. It is separate from DROID, separate from upstream
HIL-SERL examples, and separate from the legacy RealSense-style
`plug_insertion` profile.

Current camera contract:

```text
image_keys = ["zed_left", "zed_right"]
classifier_keys = ["zed_left", "zed_right"]
image_shape = [128, 128, 3]
image_dtype = uint8
camera_service = http://127.0.0.1:54819
```

No live env construction, reset, step, FCI read, gripper command, motion
command, policy rollout, reward collection, classifier training, or checkpoint
promotion is enabled by this folder.

No-motion validation:

```bash
cd /home/robot/serl_projects/hil-serl-fr3
source env/activation.sh
python tools/no_motion_validate.py
python tools/run_plug_actor.py --dry-run --headless --camera-profile zed_stereo
python tools/run_plug_learner.py --dry-run --headless --camera-profile zed_stereo
```

The ZED service itself is started from:

```text
artifacts/zed_camera_service_runtime/scripts/run_zed_camera_service_runtime.sh
```

That service uses `robot1` Python for `pyzed.sl` and exposes frames only on
localhost. The HIL-SERL client consumes the service from the `hilserl-fr3`
environment.

Next gates:

1. Decide whether ZED-only visibility is sufficient for plug insertion or add a
   close-up wrist/fixture camera.
2. Collect real reward images and demonstrations using the selected camera
   schema.
3. Train and validate the reward classifier on the remote `fzt` training
   server.
4. Integrate live FR3 state/control only after the separate live safety gate.
