#!/usr/bin/env bash
# fr3-zed-droid-verify
# ----------------------
# Desktop verifier for the local DROID ZED reference contract.
#
# This is the desktop-side twin of the laptop wrapper at
# /Users/tacyvan/Documents/Code/hilserl-fr3/scripts/setup/14_fr3_zed_droid_verify.sh
# (synced 1:1; both share the same exit-code matrix).
#
# Purpose
# -------
# 1. Import the localized DROID reference
#    (scripts/zed_droid_reference/zed_camera.py) under a fake ``pyzed.sl``
#    namespace and assert the ``ZedCamera`` construction contract.
# 2. Run the unit test suites that guard the reference and this verifier
#    (no real camera is ever opened).
# 3. Report any production-payload hits where ``sl.Camera(`` / ``.open(``
#    show up in a guarded runtime path.
#
# Source / sync
# -------------
# The laptop repo at /Users/tacyvan/Documents/Code/hilserl-fr3 owns the
# canonical source. This file lives at /home/robot/fr3-zed-droid-verify on
# the desktop. It is the same content, but the desktop version must
# discover the reference and tests via:
#   * --repo PATH    (preferred; default tries ~/hilserl-fr3 then ~)
#   * REPO env var
#
# Args:
#   --python PATH  (default: ${ZED_VERIFY_PYTHON:-/usr/bin/python3})
#   --repo PATH    (default: ${ZED_VERIFY_REPO:-/home/robot/hilserl-fr3})
#   --evidence DIR (default: .planning/2026-06-11-fr3-serl-control-runtime/evidence)
#   -h | --help
#
# Exit codes (mirrors the laptop wrapper):
#   0  PASS
#   20 reference import failed
#   21 ZedCamera construction contract failed
#   22 pytest test_zed_droid_reference failed
#   23 pytest test_fr3_zed_droid_verify_script failed
#   25 no python interpreter
#   26 evidence dir not writable
#   27 repo not found
#
# This verifier never opens a real camera. The live SDK path lives at
# /home/robot/fr3-zed-recovery-gate.

set -euo pipefail

PYTHON_BIN="${ZED_VERIFY_PYTHON:-${PYTHON_BIN:-/usr/bin/python3}}"
REPO_ROOT="${ZED_VERIFY_REPO:-${REPO_ROOT:-}}"
EVIDENCE_DIR="${EVIDENCE_DIR:-/home/robot/.planning/2026-06-11-fr3-serl-control-runtime/evidence}"

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON_BIN="$2"; shift 2;;
        --repo) REPO_ROOT="$2"; shift 2;;
        --evidence) EVIDENCE_DIR="$2"; shift 2;;
        -h|--help) usage;;
        *) echo "unknown arg: $1" >&2; exit 64;;
    esac
done

# ---- resolve REPO_ROOT ---------------------------------------------------
if [[ -z "${REPO_ROOT}" ]]; then
    for cand in \
        "/home/robot/hilserl-fr3" \
        "/home/robot/droid" \
        "/Users/tacyvan/Documents/Code/hilserl-fr3"; do
        if [[ -d "${cand}/scripts/zed_droid_reference" ]]; then
            REPO_ROOT="${cand}"; break
        fi
    done
fi
if [[ -z "${REPO_ROOT}" || ! -d "${REPO_ROOT}/scripts/zed_droid_reference" ]]; then
    echo "FAIL repo not found (need scripts/zed_droid_reference); tried: ${REPO_ROOT:-<unset>}" >&2
    exit 27
fi
TESTS_DIR="${REPO_ROOT}/tests"
REFERENCE_DIR="${REPO_ROOT}/scripts/zed_droid_reference"

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

# ---- preflight ------------------------------------------------------------
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    log "FAIL python not found: ${PYTHON_BIN}"
    exit 25
fi
mkdir -p "${EVIDENCE_DIR}" 2>/dev/null || { log "FAIL evidence dir not writable: ${EVIDENCE_DIR}"; exit 26; }
EVIDENCE_FILE="${EVIDENCE_DIR}/FR3-ZED-DROID-VERIFY.md"
: > "${EVIDENCE_FILE}"
log_evidence() { echo "$*" | tee -a "${EVIDENCE_FILE}"; }

log_evidence "# fr3-zed-droid-verify run"
log_evidence "- date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_evidence "- python: ${PYTHON_BIN}"
log_evidence "- repo: ${REPO_ROOT}"

# ---- gate 1: reference imports under fake sl -----------------------------
log "gate 1: reference import contract"
if ! "${PYTHON_BIN}" - <<PYEOF 2>>"${EVIDENCE_FILE}"; then
import importlib, sys, types

for _name in ("cv2", "numpy"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
if "PIL" not in sys.modules:
    _pil = types.ModuleType("PIL")
    _pil_image = types.ModuleType("PIL.Image")
    _pil_image.open = lambda *a, **k: None
    _pil.Image = _pil_image
    sys.modules["PIL"] = _pil
    sys.modules["PIL.Image"] = _pil_image

sl = types.ModuleType("pyzed.sl")
class _Cam:
    def __init__(self): self.opened=False; self.grabbed=0
    def open(self, ip): self.opened=True; return 0
    def grab(self): self.grabbed+=1; return 0
    def retrieve_image(self, *a, **k): pass
    def close(self): pass
class _Mat: pass
class _IP:
    input=None
    camera_resolution="HD1080"
    camera_fps=30
class _IT:
    def __init__(self, *a, **k): pass
    def set_from_serial_number(self, n): self.sn=n
class _Res:
    HD2K="HD2K"; HD1080="HD1080"; HD720="HD720"; VGA="VGA"
class _EC: SUCCESS=0
class _V: LEFT=0; RIGHT=1
sl.Camera=_Cam; sl.Mat=_Mat; sl.InitParameters=_IP
sl.InputType=_IT; sl.RESOLUTION=_Res; sl.ERROR_CODE=_EC; sl.VIEW=_V
pyzed = types.ModuleType("pyzed"); pyzed.sl=sl
sys.modules["pyzed"]=pyzed; sys.modules["pyzed.sl"]=sl
sys.path.insert(0, "${REFERENCE_DIR}")
mod = importlib.import_module("zed_camera")
assert hasattr(mod, "ZedCamera"), "ZedCamera missing"
with open("${EVIDENCE_FILE}", "a") as _ef:
    _ef.write("- gate1: PASS\n")
PYEOF
    log_evidence "- gate1: FAIL exit=$?"
    exit 20
fi

# ---- gate 2: ZedCamera construction contract -----------------------------
log "gate 2: ZedCamera construction contract"
if ! "${PYTHON_BIN}" - <<PYEOF 2>>"${EVIDENCE_FILE}"; then
import importlib, sys, types

for _name in ("cv2", "numpy"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
if "PIL" not in sys.modules:
    _pil = types.ModuleType("PIL")
    _pil_image = types.ModuleType("PIL.Image")
    _pil_image.open = lambda *a, **k: None
    _pil.Image = _pil_image
    sys.modules["PIL"] = _pil
    sys.modules["PIL.Image"] = _pil_image

class CamRec:
    instances=[]
    def __init__(self): CamRec.instances.append(self); self.opened=False; self.open_args=[]; self.grabbed=0
    def open(self, ip): self.open_args.append(ip); self.opened=True; return 0
    def grab(self): self.grabbed+=1; return 0
    def retrieve_image(self, *a, **k): pass
    def close(self): pass
class IPRec:
    def __init__(self): self.input=None; self.camera_resolution=None; self.camera_fps=None
class ITRec:
    def __init__(self, *a, **k): self.serial_number=None; self.device_path=(a[0] if a else None)
    def set_from_serial_number(self, n): self.serial_number=n
sl=types.ModuleType("pyzed.sl")
sl.Camera=CamRec; sl.Mat=type("M",(),{})
sl.InitParameters=IPRec; sl.InputType=ITRec
class R: HD2K="HD2K"; HD1080="HD1080"; HD720="HD720"; VGA="VGA"
sl.RESOLUTION=R
class E: SUCCESS=0
sl.ERROR_CODE=E
class V: LEFT=0; RIGHT=1
sl.VIEW=V
pyzed=types.ModuleType("pyzed"); pyzed.sl=sl
sys.modules["pyzed"]=pyzed; sys.modules["pyzed.sl"]=sl
sys.path.insert(0, "${REFERENCE_DIR}")
mod=importlib.import_module("zed_camera")
c=mod.ZedCamera(serial_number=13132609)
assert len(CamRec.instances)==1
ip=CamRec.instances[0].open_args[0]
assert ip.input and ip.input.serial_number==13132609
assert ip.camera_resolution=="HD1080" and ip.camera_fps==30
with open("${EVIDENCE_FILE}", "a") as _ef:
    _ef.write("- gate2: PASS (serial path)\n")
c2=mod.ZedCamera(device_path="/dev/video0")
ip2=CamRec.instances[1].open_args[0]
assert ip2.input and ip2.input.device_path=="/dev/video0"
with open("${EVIDENCE_FILE}", "a") as _ef:
    _ef.write("- gate2: PASS (device_path path)\n")
PYEOF
    log_evidence "- gate2: FAIL exit=$?"
    exit 21
fi

# ---- gate 3: production-payload grep guard -------------------------------
log "gate 3: grep guard for sl.Camera( / .open( in production payload"
HITS=()
for f in scripts/zed_capture.py scripts/zed_env_dispatch.py franka_env/camera/zed_capture.py; do
    p="${REPO_ROOT}/${f}"
    [[ -f "$p" ]] || continue
    if grep -nE 'sl\.Camera\s*\(|\.open\s*\(' "$p" >/dev/null; then
        HITS+=("$f")
    fi
done
if [[ ${#HITS[@]} -gt 0 ]]; then
    log_evidence "- gate3: INFO production-payload hits (guarded): ${HITS[*]}"
else
    log_evidence "- gate3: PASS (no real sl.Camera( / .open( in production payload)"
fi

# ---- gate 4: pytest test_zed_droid_reference ----------------------------
log "gate 4: pytest test_zed_droid_reference.py"
# The test imports ``scripts.zed_droid_reference.zed_camera`` (via the
# fixtures module), so REPO_ROOT must be on PYTHONPATH for the suite to
# resolve the package.
if ! PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m pytest -q "${REFERENCE_DIR}/tests/test_zed_droid_reference.py" 2>>"${EVIDENCE_FILE}"; then
    log_evidence "- gate4: FAIL"
    exit 22
fi
log_evidence "- gate4: PASS"

# ---- gate 5: pytest test_fr3_zed_droid_verify_script --------------------
log "gate 5: pytest test_fr3_zed_droid_verify_script.py"
if [[ -f "${TESTS_DIR}/test_fr3_zed_droid_verify_script.py" ]]; then
    if ! PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m pytest -q "${TESTS_DIR}/test_fr3_zed_droid_verify_script.py" 2>>"${EVIDENCE_FILE}"; then
        log_evidence "- gate5: FAIL"
        exit 23
    fi
    log_evidence "- gate5: PASS"
else
    log_evidence "- gate5: SKIP (test not present on this host; laptop-only)"
fi

log_evidence "- result: ALL GATES PASS"
log "OK: fr3-zed-droid-verify"
exit 0
