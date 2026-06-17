# Plug Insertion

Owner: Codex primary, Human approval required before live robot use

This task folder is reserved for plug-style fine insertion. It starts from the
upstream USB insertion structure but remains isolated from DROID and other HIL
experiments.

Runtime imports for the local plug experiment now use the non-colliding package
`fr3_experiments.plug_insertion`. The older `experiments/plug_insertion/config.py`
file is retained as historical context only, because upstream HIL-SERL already
owns the `experiments` package name under `upstream/hil-serl/examples`.

Gate 1 no-motion validation must pass before any live backend is selected:

```bash
cd /home/robot/serl_projects/hil-serl-fr3
source env/activation.sh
python tools/no_motion_validate.py
```

The validator must import `fr3_experiments.plug_insertion.config`, construct the
plug fake env under the localhost no-motion mock server, exercise the 7D action
validation matrix, reject old 8D OpenPI/DROID passthrough actions, run the
static/runtime DROID guard, and validate image/camera-key handling.

No live command should be run from this folder until the bridge backend and live
robot gate are separately approved.
