#!/usr/bin/env bash
set -euo pipefail

source /home/robot/serl_projects/hil-serl-fr3/env/activation.sh

if [[ "${SERL_FR3_NO_MOTION:-}" != "1" ]]; then
  echo "HEADLESS_RUNNER_NO_MOTION_ONLY" >&2
  exit 64
fi

export PYNPUT_BACKEND=dummy
exec "$@"
