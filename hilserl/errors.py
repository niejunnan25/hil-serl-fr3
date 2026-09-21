"""Device errors whose recovery requires a new operator reset confirmation."""
from __future__ import annotations

import copy


class RobotStateUnavailable(RuntimeError):
    """Stop issuing actions, retain the Actor, and await an explicit new reset.

    A timed out command may already have reached the robot. Callers must never
    retry it automatically, including when state feedback subsequently returns.
    """

    def __init__(self, message, *, endpoint="getstate", status_code=None,
                 details=None, health=None, delivery_unknown=False):
        super().__init__(message)
        self.endpoint = endpoint
        self.status_code = status_code
        self.details = copy.deepcopy(details or {})
        self.health = copy.deepcopy(health or {})
        self.delivery_unknown = bool(delivery_unknown)

    def as_dict(self):
        return dict(type=type(self).__name__, message=str(self), endpoint=self.endpoint,
                    status_code=self.status_code, details=copy.deepcopy(self.details),
                    health=copy.deepcopy(self.health), delivery_unknown=self.delivery_unknown)
