"""Image profile identifiers and their naming scheme.

Kept free of numpy/cv2 on purpose: `hilserl.config` validates profile names, and
`bin/hil-serl` must stay importable by an interpreter without the camera stack.
Pixel geometry lives in `hilserl.image_profile`.
"""
from __future__ import annotations

FULL_FRAME = "full-frame128-v1"
INSERT_ROI = "insert-roi224-v1"
FRONT_ROI160 = "insert-front-roi160-v1"
POLICY_SIZES = (128, 160, 192, 224)

# One family per physical camera mount. A moved camera gets a new family name
# instead of edited numbers under the existing one: seed manifests pin the whole
# profile dict, so an in-place edit would fail their validation.
PROFILE_NAMES = (FULL_FRAME,
                 *(f"insert-roi{size}-v1" for size in POLICY_SIZES),
                 *(f"insert-front-roi{size}-v1" for size in POLICY_SIZES))
