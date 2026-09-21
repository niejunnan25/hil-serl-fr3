"""Archive contract tests use synthetic arrays and never access robot or GPU I/O."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from hilserl.control import OperatorControl, StopRequested, send_command
from hilserl.episodes import run_episodes
from hilserl.storage import SessionRecorder, read_step, RecordingError, stamp
from scripts.video_capture import VideoCapture


class FakeEnvironment:
    def __init__(self, terminal_at=3, recorder=None):
        self.unwrapped = self
        self.frame_references = {}
        self.terminal_at = terminal_at
        self.steps = 0
        self.resets = 0
        self.closed = False
        self.recorder = recorder

    def observation(self):
        return {"state": np.full((1, 19), self.steps, dtype=np.float32),
                "side_policy": np.full((1, 8, 8, 3), self.steps, dtype=np.uint8)}

    def raw_state(self):
        return {"values": {"q": np.arange(7, dtype=np.float64)}, "capture": stamp()}

    def reset(self):
        self.resets += 1
        self.steps = 0
        return self.observation(), {"reset": {"success": True, "synthetic": True}}

    def step(self, action):
        if self.recorder:
            self.recorder.check()
        self.steps += 1
        done = self.steps >= self.terminal_at
        info = {"termination_reason": "classifier" if done else None,
                "controller_input_action": action.copy(),
                "controller_commands": [{"kind": "pose", "pose": np.arange(7, dtype=np.float32)}]}
        return self.observation(), int(done), done, False, info

    def close(self):
        self.closed = True


class ScriptedOperator:
    def __init__(self, *, outcome=0, on_wait=None, label_after=None):
        self.state = {"phase": "starting", "episode_id": None, "step": 0}
        self.outcome, self.on_wait, self.label_after = outcome, on_wait, label_after

    def publish(self, phase=None, **fields):
        if phase:
            self.state["phase"] = phase
        self.state.update(fields)

    def raise_if_stop(self):
        pass

    def poll(self, allowed):
        if self.label_after is not None and self.state["step"] >= self.label_after:
            return {"command": "label", "value": self.outcome}
        return None

    def wait(self, phase, prompt, allowed, check=None):
        self.publish(phase)
        if check:
            check()
        if self.on_wait:
            self.on_wait(phase)
        return {"command": "label", "value": self.outcome} if phase == "awaiting_label" else {"command": "continue"}


class SessionRecordingTests(unittest.TestCase):
    def test_terminal_step_is_durable_before_manual_override_and_wait_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "recording"
            recorder = SessionRecorder(root, min_free_bytes=0)
            env = FakeEnvironment(recorder=recorder)
            seen = []

            def waiting(phase):
                if phase == "awaiting_label":
                    paths = list((root / "episodes/000001").glob("*.npz"))
                    self.assertEqual(len(paths), 3)
                    self.assertTrue(read_step(sorted(paths)[-1])["terminated"])
                    time.sleep(0.1)

            operator = ScriptedOperator(outcome=0, on_wait=waiting)
            outcomes = run_episodes(env, lambda obs, n: (np.arange(7), {"revision": n}), recorder, operator,
                                    max_episodes=1, emit=seen.append)
            self.assertEqual(env.steps, 3)
            self.assertEqual(env.resets, 1)
            self.assertEqual(len(seen), 3)
            self.assertEqual(seen[-1]["rewards"], 0)
            self.assertTrue(seen[-1]["dones"])
            self.assertTrue(seen[-1]["infos"]["manual_success"])
            self.assertLess(outcomes[0]["collection_seconds"], .1)
            metadata = json.loads((root / "episodes/000001/episode.json").read_text())
            self.assertTrue(metadata["complete"])
            self.assertEqual(metadata["outcome"], 0)
            raw = read_step(root / "episodes/000001/000002.npz")
            self.assertEqual(raw["observed_reward"], 1)
            np.testing.assert_array_equal(raw["proposed_action"], np.arange(7))
            np.testing.assert_array_equal(raw["observations"]["state"], np.full((1, 19), 2))

    def test_manual_label_between_ticks_does_not_send_an_extra_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = SessionRecorder(Path(temporary) / "recording", min_free_bytes=0)
            env = FakeEnvironment(terminal_at=100, recorder=recorder)
            seen = []
            run_episodes(env, lambda obs, n: (np.zeros(7), {}), recorder,
                         ScriptedOperator(outcome=1, label_after=2), max_episodes=1, emit=seen.append)
            self.assertEqual(env.steps, 2)
            self.assertEqual(len(seen), 2)
            self.assertEqual(seen[-1]["infos"]["termination_reason"], "manual")

    def test_timeout_includes_last_step_and_operator_stop_preserves_unlabeled_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "recording"
            recorder = SessionRecorder(root, min_free_bytes=0)
            env = FakeEnvironment(terminal_at=100, recorder=recorder)

            def stop(phase):
                if phase == "awaiting_label":
                    raise StopRequested()

            run_episodes(env, lambda obs, n: (np.zeros(7), {}), recorder,
                         ScriptedOperator(on_wait=stop), max_episode_steps=2, max_episodes=1)
            self.assertEqual(env.steps, 2)
            self.assertEqual(len(list((root / "episodes/000001").glob("*.npz"))), 2)
            metadata = json.loads((root / "episodes/000001/episode.json").read_text())
            self.assertFalse(metadata["complete"])
            self.assertIsNone(metadata["outcome"])
            self.assertEqual(metadata["termination_reason"], "time_limit")
            self.assertTrue(env.closed)

    def test_writer_failure_latches_fault_and_prevents_further_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = SessionRecorder(Path(temporary) / "recording", min_free_bytes=0)
            env = FakeEnvironment(terminal_at=1, recorder=recorder)
            with patch("hilserl.storage.write_step", side_effect=OSError("disk test failure")):
                with self.assertRaises(RecordingError):
                    run_episodes(env, lambda obs, n: (np.zeros(7), {}), recorder,
                                 ScriptedOperator(), max_episodes=1)
            self.assertEqual(env.steps, 1)
            self.assertEqual(env.resets, 1)
            self.assertIn("disk test failure", recorder.error)

    def test_late_or_duplicate_commands_cannot_label_another_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            control = OperatorControl(temporary, terminal=False)
            control.publish("collecting", episode_id="000001")
            send_command(temporary, "label", 1)
            control.publish("collecting", episode_id="000002")
            self.assertIsNone(control.poll({"label"}))
            send_command(temporary, "label", 0)
            self.assertEqual(control.poll({"label"})["value"], 0)
            self.assertIsNone(control.poll({"label"}))
            control.publish("resetting")
            with self.assertRaises(ValueError):
                send_command(temporary, "continue")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_continuous_video_runs_during_wait_and_has_single_camera_owner(self):
        class Camera:
            name = "side_policy"
            def __init__(self):
                self.owners, self.n = set(), 0
            def read(self):
                self.owners.add(threading.get_ident())
                time.sleep(1 / 30)
                self.n += 1
                return True, np.full((32, 64, 3), self.n % 255, np.uint8)
            def close(self):
                self.owners.add(threading.get_ident())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "recording"
            recorder = SessionRecorder(root, min_free_bytes=0, segment_seconds=1)
            source = Camera()
            camera = VideoCapture(source, recorder=recorder, continuous=True)
            a = camera.read()
            before = camera.last_read["capture_id"]
            # No env.step/read calls here: simulates waiting for the operator/reset.
            # Segment boundaries follow encoded frame count, not wall time.
            # Scheduler/encoder startup can produce fewer than 30 frames in 1.15 s.
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                recorder.check()
                writer = recorder.videos.get("side_policy")
                if writer and writer.frames_written >= recorder.fps + 3:
                    break
                time.sleep(.01)
            b = camera.read()
            self.assertGreater(camera.last_read["capture_id"], before + 20)
            camera.close()
            recorder.close()
            self.assertIsNone(recorder.error)
            self.assertEqual(len(source.owners), 1)
            movies = sorted((root / "video/side_policy").glob("*.mp4"))
            self.assertGreaterEqual(len(movies), 2)
            total = 0
            for movie in movies:
                output = subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                         "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", str(movie)])
                stream = json.loads(output)["streams"][0]
                self.assertEqual((stream["width"], stream["height"]), (64, 32))
                total += int(stream["nb_read_frames"])
            self.assertEqual(total, recorder.videos["side_policy"].frames_written)


if __name__ == "__main__":
    unittest.main()
