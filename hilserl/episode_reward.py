"""Actor-side reward worker. Device calls remain on the Actor's main thread."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import pickle
import threading
import time
import uuid

from hilserl.control import StopRequested
from hilserl.episode_commit import REQUEST, envelope, episode_key, save_json, save_pickle
from hilserl.reward_provider import relabel


class RewardTransport:
    """One worker owns one socket for its whole lifetime."""
    def __init__(self, ip, port):
        from agentlace.zmq_wrapper.req_rep import ReqRepClient
        self.client = ReqRepClient(ip, port, timeout_ms=5000)

    def request(self, value):
        return self.client.send_msg(dict(type=REQUEST, payload=value))

    def close(self):
        self.client.socket.close(linger=0)
        self.client.context.term()


class EpisodeRewardPipeline:
    def __init__(self, spec, directory, status_path, run_id, actor_attempt, provider_factory,
                 transport_factory, *, timeout=120.0):
        self.spec, self.run_id, self.actor_attempt = spec, run_id, actor_attempt
        self.directory, self.status_path = Path(directory), Path(status_path)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.provider_factory, self.transport_factory = provider_factory, transport_factory
        self.timeout = timeout
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.transitions = []
        self.entry = None
        self.error = None
        self.state = dict(phase="idle", actor_attempt=actor_attempt, contract_sha256=spec.sha256)
        self._publish()

    def _publish(self, **fields):
        with self.lock:
            self.state.update(fields, monotonic_ns=time.monotonic_ns(), unix_ns=time.time_ns())
            save_json(self.status_path, self.state)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def check(self):
        if self.error:
            raise RuntimeError(f"Reward episode retained for recovery: {self.error}")

    def _check_cancel(self):
        if self.cancel.is_set():
            raise StopRequested("Reward work cancelled; pending data retained")

    def begin(self, episode_id):
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Previous reward gap has not been released")
        self.transitions = []
        self.entry = dict(source_attempt=self.actor_attempt, episode_id=episode_id)
        self.error = None
        self._publish(phase="collecting", episode_id=episode_id, steps=0, error=None,
                      receipt=None, extra_wait_seconds=0, inference_seconds=None)

    def append(self, transition, raw):
        if not self.entry:
            raise RuntimeError("No reward episode")
        value = copy.deepcopy(transition)
        # Keep exact capture/command context in the durable pending payload.
        value["infos"]["capture"] = {key: copy.deepcopy(raw.get(key)) for key in
            ("step_started", "step_ended", "frame_references", "next_frame_references")}
        self.transitions.append(value)
        with self.lock:
            self.state["steps"] = len(self.transitions)

    def end_collection(self):
        if not self.transitions:
            return
        self._start(self.transitions, self.entry)

    def _start(self, raw, entry, *, finalized=None):
        self.entry = entry
        self.entry_dir = self.directory / episode_key(entry["source_attempt"], entry["episode_id"])
        self.entry_dir.mkdir(parents=True, exist_ok=True)
        self.label_ready = threading.Event()
        self.release_requested = threading.Event()
        self.committed = threading.Event()
        self.done = threading.Event()
        self.finalized = finalized
        self.error = None
        self.gap_id = uuid.uuid4().hex
        save_json(self.entry_dir / "entry.json", dict(entry, run_id=self.run_id,
                  contract_sha256=self.spec.sha256))
        if finalized is None:
            save_pickle(self.entry_dir / "raw.pkl", raw)
        if finalized is not None:
            self.label_ready.set()
        self._publish(phase="waiting_learner", episode_id=entry["episode_id"], gap_id=self.gap_id,
                      steps=len(raw), gap_started_monotonic_ns=time.monotonic_ns(), error=None,
                      receipt=None)
        self.thread = threading.Thread(target=self._worker, args=(raw,), name="episode-reward", daemon=True)
        self.thread.start()

    def finalize(self, outcome, termination_reason):
        if not self.transitions:
            return
        if not self.thread:
            raise RuntimeError("Reward scoring did not start")
        finalized = copy.deepcopy(self.transitions)
        final = finalized[-1]
        final.update(rewards=float(outcome), dones=True, masks=0.0)
        final["infos"].update(succeed=bool(outcome), manual_success=True, verdict_source="human",
                              termination_reason=termination_reason)
        save_pickle(self.entry_dir / "labeled.pkl", finalized)
        (self.entry_dir / "raw.pkl").unlink(missing_ok=True)
        self.finalized = finalized
        self.label_ready.set()

    def _request(self, transport, operation, *, generation=None, **fields):
        request = dict(operation=operation, run_id=self.run_id, actor_attempt=self.actor_attempt,
                       contract_sha256=self.spec.sha256, gap_id=self.gap_id,
                       generation=generation, **fields)
        deadline = time.monotonic() + self.timeout
        while True:
            self._check_cancel()
            result = transport.request(request)
            if result and result.get("success"):
                return result
            if result and result.get("stale_generation"):
                raise RuntimeError("Learner restarted during reward gap; recover pending data with a new Actor")
            if result and result.get("error"):
                raise RuntimeError(result["error"])
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Reward {operation} acknowledgment timed out")
            self.cancel.wait(0.05)

    def _worker(self, raw):
        transport = provider = None
        try:
            transport = self.transport_factory()
            response = self._request(transport, "begin", **self.entry)
            generation = response["generation"]
            deadline = time.monotonic() + self.timeout
            while not response["paused"]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Learner did not finish its current update group")
                self._check_cancel(); self.cancel.wait(0.05)
                response = self._request(transport, "status", generation=generation)
            self._publish(phase="scoring", learner_generation=generation,
                          paused_at_update=response["last_update"], scoring_started_monotonic_ns=time.monotonic_ns())
            scored_path = self.entry_dir / "scores.json"
            if scored_path.exists():
                scored = json.loads(scored_path.read_text())
            else:
                provider = self.provider_factory()
                scored = provider.score(raw, check=self._check_cancel)
                self._check_cancel()
                save_json(scored_path, scored)
            self._publish(phase="awaiting_label", inference_seconds=scored["inference_seconds"],
                          scoring_finished_monotonic_ns=time.monotonic_ns())
            while not self.label_ready.wait(0.05):
                self._check_cancel()
            self._check_cancel()
            labeled = relabel(self.finalized, scored, self.spec)
            value = envelope(self.run_id, self.entry["source_attempt"], self.entry["episode_id"],
                             self.finalized, labeled, self.spec)
            save_json(self.entry_dir / "ready.json", dict(payload_digest=value["payload_digest"],
                raw_digest=value["raw_digest"], contract_sha256=self.spec.sha256,
                transitions=[dict(raw_transition_id=t["infos"]["raw_transition_id"],
                                  reward=t["rewards"], **t["infos"]["reward"]) for t in labeled]))
            self._publish(phase="committing", commit_started_monotonic_ns=time.monotonic_ns())
            receipt = self._request(transport, "commit", generation=generation, episode=value)["receipt"]
            if (receipt.get("generation") != generation or receipt.get("payload_digest") != value["payload_digest"]
                    or receipt.get("online_count") != len(labeled)
                    or receipt.get("contract_sha256") != self.spec.sha256):
                raise ValueError("Incomplete or wrong episode commit receipt")
            save_json(self.entry_dir / "receipt.json", receipt)
            self._publish(phase="committed", receipt=receipt, committed_monotonic_ns=time.monotonic_ns())
            self.committed.set()
            while not self.release_requested.wait(0.05):
                self._check_cancel()
            self._request(transport, "release", generation=generation, receipt=receipt)
            save_json(self.entry_dir / "released.json", dict(receipt=receipt, unix_ns=time.time_ns()))
            # The receiver has the complete durable transaction and the raw
            # recorder retains the source. Keep compact sidecars, not another
            # permanent copy of every camera array in the pending directory.
            (self.entry_dir / "labeled.pkl").unlink(missing_ok=True)
            self._publish(phase="released", released_monotonic_ns=time.monotonic_ns())
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._publish(phase="pending_recovery", error=self.error)
        finally:
            try:
                if provider is not None:
                    provider.close()
            finally:
                try:
                    if transport is not None:
                        transport.close()
                finally:
                    self.done.set()

    def barrier(self, operator, recorder, *, final=False, check=lambda: None):
        """Wait after reset motion, before its ready cue; keep checking the hold."""
        def check_wait():
            operator.raise_if_stop(); recorder.check(); self.check()
            check()
            operator.raise_if_stop()

        check_wait()
        if not self.thread:
            return
        started = time.monotonic()
        operator.publish("draining_reward" if final else "waiting_reward",
                         prompt="等待本条奖励计算和完整入库，录像与墙钟计时继续。")
        while not self.committed.is_set():
            check_wait()
            operator.publish(reward=self.snapshot())
            self.committed.wait(0.05)
        check_wait()
        self.release_requested.set()
        while not self.done.wait(0.05):
            check_wait()
            operator.publish(reward=self.snapshot())
        check_wait()
        self.thread.join()
        elapsed = time.monotonic() - started
        self._publish(extra_wait_seconds=elapsed, final_drain=final)
        recorder.event("reward_barrier_complete", reward=self.snapshot())
        operator.publish(reward=self.snapshot())
        self.thread = None
        self.transitions = []

    def recover(self, operator, recorder):
        """Finish labeled pending data without replaying any robot commands."""
        for directory in sorted(self.directory.iterdir()):
            if not directory.is_dir() or (directory / "released.json").exists():
                continue
            label = directory / "labeled.pkl"
            if not label.exists():
                recorder.event("reward_recovery_skipped", directory=str(directory), reason="no_complete_human_label")
                continue
            entry = json.loads((directory / "entry.json").read_text())
            if entry.get("run_id") != self.run_id or entry.get("contract_sha256") != self.spec.sha256:
                raise ValueError("Pending episode belongs to a different run/reward contract")
            with label.open("rb") as stream:
                raw = pickle.load(stream)
            self._start(raw, {k: entry[k] for k in ("source_attempt", "episode_id")}, finalized=raw)
            self.barrier(operator, recorder)

    def abort(self, reason):
        if self.transitions and self.entry and not self.thread:
            directory = self.directory / episode_key(self.entry["source_attempt"], self.entry["episode_id"])
            save_pickle(directory / "incomplete.pkl", self.transitions)
            save_json(directory / "incomplete.json", dict(reason=reason, source=self.entry))
            self._publish(phase="incomplete", reason=reason)
        self.transitions = []

    def close(self):
        self.cancel.set()
        if self.thread:
            # RPC/model calls have explicit timeouts. Never delay robot close
            # for a whole inference timeout; the durable pending payload remains.
            self.thread.join(timeout=0.2)
