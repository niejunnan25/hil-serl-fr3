"""Canonical import path for the ZED camera adapter.

The implementation lives in `scripts.zed_capture` for CLI compatibility. Import
`ZEDCapture` from this module in runtime code that expects a SERL-style camera
package path.
"""

from scripts.zed_capture import ZEDCapture

__all__ = ["ZEDCapture"]

