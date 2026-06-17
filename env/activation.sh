#!/usr/bin/env bash
set -euo pipefail

export SERL_FR3_ROOT=/home/robot/serl_projects/hil-serl-fr3
export SERL_FR3_ENV=/home/robot/miniconda3/envs/hilserl-fr3
export SERL_FR3_ROBOT_PORT=5017
export SERL_FR3_ROS_MASTER_PORT=11317
export SERL_FR3_AGENT_PORT_BASE=54817
export SERL_FR3_NO_MOTION=1
export PYTHONDONTWRITEBYTECODE=1
export ROS_MASTER_URI=http://127.0.0.1:${SERL_FR3_ROS_MASTER_PORT}
export PYTHONPATH="${SERL_FR3_ROOT}/upstream/hil-serl/serl_robot_infra:${SERL_FR3_ROOT}/upstream/hil-serl/examples:${PYTHONPATH:-}"

source /home/robot/miniconda3/etc/profile.d/conda.sh
conda activate hilserl-fr3

_serl_fr3_site_packages="$("${SERL_FR3_ENV}/bin/python" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
export CUDA_ROOT="${_serl_fr3_site_packages}/nvidia/cuda_nvcc"
unset _serl_fr3_site_packages

cd "${SERL_FR3_ROOT}"
