"""Synthetic integration sample. Does not import any robot or training library."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hilserl.storage import SessionRecorder
from hilserl.episodes import run_episodes
from scripts.video_capture import VideoCapture
from test_session_recording import FakeEnvironment, ScriptedOperator


def create(output):
    root = Path(output)
    capture_dir = root / "synthetic_validation" / "recordings" / "sample"
    recorder = SessionRecorder(capture_dir, metadata={"mode": "train", "synthetic": True},
                               min_free_bytes=0, segment_seconds=3)
    phase = {"name": "waiting"}

    class Camera:
        def __init__(self, name):
            self.name, self.n = name, 0
        def read(self):
            time.sleep(1/30)
            frame = np.full((720, 1280, 3), (46, 53, 57), np.uint8)
            x = 70 + self.n * 9 % 1000
            frame[280:410, x:x+80] = (160, 185, 110) if self.name == "wrist_1" else (190, 145, 85)
            cv2.putText(frame, "SYNTHETIC / " + self.name, (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (230, 235, 235), 2)
            cv2.putText(frame, phase["name"] + " / frame " + str(self.n), (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 208, 210), 2)
            self.n += 1
            return True, frame
        def close(self):
            pass

    cameras = {name: VideoCapture(Camera(name), name=name, recorder=recorder, continuous=True)
               for name in ("wrist_1", "side_policy")}
    class Environment(FakeEnvironment):
        def reset(self):
            phase["name"] = "RESET"
            time.sleep(.5)
            return super().reset()
        def step(self, action):
            phase["name"] = "COLLECT"
            time.sleep(.15)
            result = super().step(action)
            if self.steps == 2:
                result[-1]["intervene_action"] = np.zeros(7, np.float32)
            self.frame_references = {name: camera.last_read for name, camera in cameras.items()}
            for camera in cameras.values():
                camera.read()
            return result
        def close(self):
            for camera in cameras.values():
                camera.close()
            super().close()
    class Operator(ScriptedOperator):
        labels = [1, 0]
        def wait(self, state, prompt, allowed, check=None):
            phase["name"] = state.upper()
            time.sleep(.6)
            self.outcome = self.labels.pop(0) if state == "awaiting_label" else 0
            return super().wait(state, prompt, allowed, check)
    for camera in cameras.values():
        camera.read()
    result = run_episodes(Environment(recorder=recorder), lambda obs,n: (np.arange(7)/20, {"revision": 1}),
                          recorder, Operator(), max_episodes=2)
    print(json.dumps({"directory":str(capture_dir), "episodes":result, "error":recorder.error}))
    if recorder.error or len(result) != 2:
        raise RuntimeError("Synthetic integration did not finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    create(parser.parse_args().output)
