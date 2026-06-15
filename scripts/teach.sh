#!/usr/bin/env bash
# teach.sh — run an interactive FR3 GELLO/Xbox demo-collection session on the
# desktop. Resets the robot to HOME, starts the hybrid teleop recording, and on
# Ctrl-C prompts success/fail and logs the demo.
#
#   ./scripts/teach.sh              # reset to HOME, then record
#   ./scripts/teach.sh --no-reset   # skip the home reset
#   ./scripts/teach.sh --help       # all options
set -euo pipefail
REPO="${REPO:-/home/robot/hilserl-fr3}"
PY="${PY:-/home/robot/miniconda3/envs/hilserl-fr3/bin/python}"
cd "$REPO"
exec env PYTHONPATH=scripts "$PY" scripts/teach_session.py "$@"
