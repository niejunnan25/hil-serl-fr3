"""Templated obs dict that enforces 100% schema match with OpenPI policy.

Per spec §3.3 Phase 0 hard gate: any key missing / shape wrong / dtype wrong
is a hard fail; no silent fallback.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from droid.sim.obs.schema_audit import ObsKeySpec, SchemaAudit


class ObsDictMismatch(RuntimeError):
    """Raised when observation dict does not match audited schema."""


@dataclasses.dataclass(frozen=True)
class ObsDictTemplate:
    """Schema-aware validator for sim runner output dict.

    Use:
        template = build_obs_dict_template(schema_audit)
        obs = sim_runner.get_observation()
        template.validate(obs)   # raises ObsDictMismatch if wrong
    """
    expected: tuple[ObsKeySpec, ...]

    def validate(self, obs: Mapping[str, Any]) -> None:
        # 1. all expected keys present
        expected_names = {spec.name for spec in self.expected}
        actual_names = set(obs.keys())
        missing = expected_names - actual_names
        if missing:
            raise ObsDictMismatch(
                f"missing keys in obs dict: {sorted(missing)}; "
                f"got {sorted(actual_names)}"
            )

        # 2. shape + dtype match per key
        for spec in self.expected:
            value = obs[spec.name]
            actual_shape = tuple(getattr(value, "shape", ()))
            actual_dtype = str(getattr(value, "dtype", "unknown"))
            if actual_shape != spec.shape:
                raise ObsDictMismatch(
                    f"key {spec.name!r} shape mismatch: expected {spec.shape}, "
                    f"got {actual_shape}"
                )
            # Soft-match dtype (e.g., torch.uint8 vs np.uint8 vs 'uint8')
            if spec.dtype not in actual_dtype:
                raise ObsDictMismatch(
                    f"key {spec.name!r} dtype mismatch: expected {spec.dtype}, "
                    f"got {actual_dtype}"
                )

    def keys(self) -> set[str]:
        return {spec.name for spec in self.expected}


def build_obs_dict_template(audit: SchemaAudit) -> ObsDictTemplate:
    return ObsDictTemplate(expected=tuple(audit.obs_keys))


__all__ = ["ObsDictTemplate", "ObsDictMismatch", "build_obs_dict_template"]
