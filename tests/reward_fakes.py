"""Small RGB episodes and deterministic progress; never imports GPU/device code."""
import copy
import numpy as np
from hilserl.reward_provider import RewardSpec, observation_queries, prefix_indices, digest

SPEC = RewardSpec(model_id="a" * 64, image_profile="full-frame128-v1")


def observation(i):
    return dict(state=np.full((1, 19), i / 100, np.float32),
                side_policy=np.full((1, 128, 128, 3), i, np.uint8),
                wrist_1=np.full((1, 128, 128, 3), i, np.uint8))


def episode(n=3, *, outcome=0, prefix="run/actor/000001"):
    result = []
    for i in range(n):
        terminal = i == n - 1
        result.append(dict(observations=observation(i + 1), next_observations=observation(i + 2),
            actions=np.zeros(3, np.float32), rewards=float(outcome if terminal else 0),
            dones=terminal, masks=0.0 if terminal else 1.0,
            infos=dict(action_contract="fixed-xyz-v1", image_profile=SPEC.image_profile,
                       raw_transition_id=f"{prefix}/{i:06d}", episode_id=prefix.rsplit("/", 1)[-1],
                       step=i, source_action="policy", manual_success=terminal,
                       succeed=bool(outcome and terminal), verdict_source="human" if terminal else None)))
    return result


class Provider:
    def __init__(self, spec=SPEC, callback=lambda: None):
        self.spec, self.callback = spec, callback

    def score(self, transitions, *, check=lambda: None):
        self.callback(); check()
        frames, pairs = observation_queries(transitions, self.spec)
        return dict(scores=[float(frame[0, 0, 0]) / 255 for frame in frames], pairs=pairs,
                    prefixes=[prefix_indices(i, self.spec.max_frames) for i in range(len(frames))],
                    image_sha256=[digest(frame) for frame in frames], inference_seconds=0.01,
                    contract_sha256=self.spec.sha256)

    def close(self):
        pass


class Store:
    def __init__(self, *, fail_at=None):
        self.data = []
        self.fail_at = fail_at

    def insert(self, value):
        if len(self.data) == self.fail_at:
            raise OSError("Injected storage failure")
        self.data.append(copy.deepcopy(value))

    def __len__(self):
        return len(self.data)


class Transport:
    def __init__(self, server, drop=()):
        self.server, self.drop = server, set(drop)

    def request(self, payload):
        value = self.server.handle(payload)
        if payload["operation"] in self.drop:
            self.drop.remove(payload["operation"])
            return None
        return value

    def close(self):
        pass
