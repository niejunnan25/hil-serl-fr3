"""Regression coverage for reset review findings; only fake devices/state."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from hilserl import reset_feedback
from scripts import video_capture
from scripts.video_capture import VideoCapture
from test_plug_insertion_reset_safety import _Clock, _make_env


@pytest.mark.parametrize('shortfall', [.0061, .0119])
def test_stalled_withdrawal_never_relaxes_clearance_to_start_horizontal_motion(shortfall):
    clock = _Clock()
    env, _ = _make_env(clock, moving=True)
    env.config.RESET_CLEAR_Z = .15
    env.currpos[2] = .1355
    original = env._send_pos_command
    def send(pose):
        assert env.last_reset_info['phase'] != 'reset_pose', 'no lateral move before withdrawal'
        original(pose)
        if env.last_reset_info['phase'] == 'clear':
            env.currpos[2] = min(env.currpos[2], .15-shortfall)
    env._send_pos_command = send
    with pytest.raises(RuntimeError, match='clear stalled'):
        env.reset()
    assert not env.last_reset_info['success']
    assert not any(p['name']=='reset_pose' for p in env.last_reset_info['phases'])
    assert clock.now < 9


def test_expired_motion_deadline_cannot_relax_clear_tolerance():
    clock = _Clock()
    env, ns = _make_env(clock)
    ns['RESET_MOTION_MAX_TIMEOUT'] = .2
    goal = env.currpos.copy()
    goal[2] += .0061
    with pytest.raises(RuntimeError, match='clear timeout'):
        env.interpolate_move(goal, timeout=8, name='clear')
    assert clock.now <= .21


@pytest.mark.parametrize('phase', ['clear', 'reset_pose'])
def test_five_mm_stable_offset_uses_common_tolerance(phase):
    clock = _Clock()
    env, _ = _make_env(clock)
    goal = env.currpos.copy()
    goal[2] += .005
    error = env.interpolate_move(goal, timeout=8, name=phase)
    env._require_reset_settled(phase, error)
    assert error[0] == pytest.approx(.005)
    assert .5 <= clock.now < 1.
    assert env.last_reset_info['phases'][-1]['completion_position_tolerance_m'] == .006


@pytest.mark.parametrize('phase', ['clear', 'reset_pose'])
def test_beyond_common_tolerance_still_rejected_by_final_gate(phase):
    env, _ = _make_env(_Clock())
    with pytest.raises(RuntimeError, match='not_converged'):
        env._require_reset_settled(phase, (.0061, 0.))


@pytest.mark.parametrize('kind', ['velocity','drift','force','rotation','gripper'])
@pytest.mark.parametrize('when', ['before_cue','during_cue','after_cue'])
def test_changed_state_prevents_ready_or_new_observation(monkeypatch, kind, when):
    from scipy.spatial.transform import Rotation
    clock = _Clock()
    env, _ = _make_env(clock, moving=True)
    env.config.RESET_CLEAR_Z = .15
    env.config.RESET_FEEDBACK = True
    cues = []
    def change():
        if kind == 'velocity': env.dq[:] = .1
        elif kind == 'drift': env.currpos[0] += .01
        elif kind == 'force': env.currforce[:] = [46,0,0]
        elif kind == 'rotation': env.currpos[3:] = Rotation.from_rotvec([.02,0,0]).as_quat()
        else: env.curr_gripper_pos = np.array([1.])
    changed = False
    def update():
        nonlocal changed
        if when == 'before_cue' and env.last_reset_info['phase'] == 'verify' and not changed:
            changed = True
            change()
    env._update_currpos = update
    def cue(*, check):
        cues.append(clock.now)
        assert when != 'before_cue'
        clock.sleep(.4)
        change()
        if when == 'during_cue':
            check()
        clock.sleep(.4)
        return dict(state='played',duration_ms=800,pulse_count=1)
    monkeypatch.setattr(reset_feedback,'rumble_once',cue)
    env._get_obs = lambda: pytest.fail('cannot open observation/episode after lost stability')
    with pytest.raises(RuntimeError):
        env.reset()
    assert not env.last_reset_info['success']
    assert len(cues) == (0 if when=='before_cue' else 1)
    assert not any(p['name']=='ready' for p in env.last_reset_info['phases'])


class FakeCondition:
    def __init__(self, clock, on_wait=lambda: None):
        self.clock, self.on_wait = clock, on_wait
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def wait(self, timeout):
        self.clock.sleep(timeout)
        self.on_wait()


def camera(clock, *, timestamp_ns=0, on_wait=lambda: None):
    cap = object.__new__(VideoCapture)
    cap.continuous = True
    cap.name = 'synthetic'
    cap._condition = FakeCondition(clock, on_wait)
    cap._stop = threading.Event()
    cap._error = None
    cap._latest = (np.full((4,4,3),7,dtype=np.uint8),dict(capture_id=1,monotonic_ns=timestamp_ns))
    cap.last_read = None
    return cap


def test_pre_boundary_frame_is_rejected_even_inside_old_two_second_stale_limit(monkeypatch):
    clock = _Clock(); clock.now = 2.
    monkeypatch.setattr(video_capture,'time',clock)
    cap = camera(clock,timestamp_ns=1_000_000_000)
    checks=[]
    with pytest.raises(RuntimeError,match='fresh reset frame'):
        cap.read_after(1_200_000_000,deadline=2.2,check=lambda: checks.append(clock.now))
    assert cap.last_read is None
    assert clock.now == pytest.approx(2.2)
    assert max(np.diff(checks)) <= .05001


def test_reader_waits_for_new_frame_and_retains_matching_id_pixels(monkeypatch):
    clock = _Clock(); clock.now = 2.
    monkeypatch.setattr(video_capture,'time',clock)
    cap = camera(clock,timestamp_ns=1_000_000_000)
    def next_frame():
        cap._latest = (np.full((4,4,3),23,dtype=np.uint8),dict(capture_id=2,monotonic_ns=clock.monotonic_ns()))
    cap._condition.on_wait = next_frame
    frame = cap.read_after(2_000_000_000,deadline=3,check=lambda:None)
    assert cap.last_read['capture_id'] == 2 and np.all(frame==23)
    assert clock.now == pytest.approx(2.05)
    frame[:] = 0
    assert np.all(cap._latest[0]==23)


@pytest.mark.parametrize('mode',['stop','camera_error','camera_closed'])
def test_camera_wait_aborts_promptly(monkeypatch, mode):
    clock = _Clock()
    monkeypatch.setattr(video_capture,'time',clock)
    cap = camera(clock)
    class StopRequested(Exception): pass
    def check():
        if clock.now >= .1:
            if mode=='stop': raise StopRequested()
            if mode=='camera_error': cap._error = RuntimeError('unplugged')
            if mode=='camera_closed': cap._stop.set()
    with pytest.raises(StopRequested if mode=='stop' else RuntimeError):
        cap.read_after(0,deadline=2,check=check)
    assert .1 <= clock.now <= .15


def add_real_get_im(env, clock, monkeypatch, caps):
    monkeypatch.setattr(video_capture,'time',clock)
    env.cap = caps
    env.frame_references = {}
    env.config.IMAGE_PROFILE = 'full-frame128-v1'
    env.config.IMAGE_CROP = {}
    env.observation_space = {'images':{key:SimpleNamespace(shape=(4,4,3)) for key in caps}}
    env.display_image = False
    def get_obs():
        images = env.get_im()  # real implementation; no physical camera
        return dict(images=images,state=env.currpos.copy())
    env._get_obs = get_obs


def test_reset_waits_for_both_cameras_after_cue_without_extra_commands(monkeypatch):
    clock = _Clock()
    env,_ = _make_env(clock,moving=True)
    env.config.RESET_CLEAR_Z=.15
    env.config.RESET_FEEDBACK=True
    front, wrist = camera(clock), camera(clock)
    for cap in (front,wrist):
        def update(cap=cap):
            cap._latest=(np.full((4,4,3),29,dtype=np.uint8),dict(capture_id=2,monotonic_ns=clock.monotonic_ns()))
        cap._condition.on_wait=update
    add_real_get_im(env,clock,monkeypatch,dict(side_policy=front,side_classifier=front,wrist_1=wrist))
    pulse_end=[]; command_counts=[]
    def cue(*,check):
        command_counts.append(len(env.calls))
        for _ in range(16):
            check(); clock.sleep(.05)
        pulse_end.append(clock.monotonic_ns())
        return dict(state='played',duration_ms=800,pulse_count=1)
    monkeypatch.setattr(reset_feedback,'rumble_once',cue)
    obs,info=env.reset()
    assert info['reset']['success'] and len(pulse_end)==1
    assert len(env.calls)==command_counts[0], 'verification must be read-only'
    refs=info['reset']['observation']['frame_references']
    assert all(ref['monotonic_ns'] > pulse_end[0] for ref in refs.values())
    assert refs['side_policy']==refs['side_classifier']
    assert all(np.all(frame==29) for frame in obs['images'].values())
    assert not hasattr(env,'_reset_image_barrier')
    assert all(phase=='resetting' for phase,_ in env.published)


def test_slow_second_camera_refreshes_the_whole_pair(monkeypatch):
    clock=_Clock()
    env,_=_make_env(clock,moving=True)
    env.config.RESET_CLEAR_Z=.15
    front,wrist=camera(clock),camera(clock)
    first_wait=[None]
    def update_front():
        front._latest=(np.full((4,4,3),11,dtype=np.uint8),dict(capture_id=2,monotonic_ns=clock.monotonic_ns()))
    def update_wrist():
        if first_wait[0] is None: first_wait[0]=clock.now
        if clock.now-first_wait[0]>.35:
            wrist._latest=(np.full((4,4,3),22,dtype=np.uint8),dict(capture_id=3,monotonic_ns=clock.monotonic_ns()))
    front._condition.on_wait=update_front
    wrist._condition.on_wait=update_wrist
    add_real_get_im(env,clock,monkeypatch,dict(side_policy=front,wrist_1=wrist))
    _,info=env.reset()
    refs=info['reset']['observation']['frame_references']
    assert info['reset']['success']
    assert max(clock.monotonic_ns()-r['monotonic_ns'] for r in refs.values()) <= 250_000_000


@pytest.mark.parametrize('mode',['stalled_camera','state_changed','stop'])
def test_fresh_frame_failure_never_reports_ready_and_always_removes_barrier(monkeypatch, mode):
    clock=_Clock()
    env,_=_make_env(clock,moving=True)
    env.config.RESET_CLEAR_Z=.15
    front,wrist=camera(clock),camera(clock)
    class StopRequested(Exception): pass
    waits=[]
    def waiting():
        waits.append(clock.now)
        if mode=='state_changed': env.dq[:]=.1
        if mode=='stop': env.operator.raise_if_stop=lambda: (_ for _ in ()).throw(StopRequested())
    front._condition.on_wait=waiting
    add_real_get_im(env,clock,monkeypatch,dict(side_policy=front,wrist_1=wrist))
    with pytest.raises(StopRequested if mode=='stop' else RuntimeError):
        env.reset()
    assert waits and not env.last_reset_info['success']
    assert not any(p['name']=='ready' for p in env.last_reset_info['phases'])
    assert not hasattr(env,'_reset_image_barrier')
    if mode=='stalled_camera': assert clock.now-waits[0] <= 2.01


# Reuse the fake evdev fixture to exercise the real pulse and its cleanup path.
from test_reset_feedback import hardware


def test_real_rumble_cleanup_runs_when_robot_loses_stability(hardware):
    clock,calls,_,_=hardware
    env,_=_make_env(clock,moving=True)
    env.config.RESET_CLEAR_Z=.15
    env.config.RESET_FEEDBACK=True
    def update():
        if any(c==('write',(21,7,1)) for c in calls):
            env.dq[:]=.1
    env._update_currpos=update
    env._get_obs=lambda: pytest.fail('must not collect after interrupted cue')
    with pytest.raises(RuntimeError,match='lost_stability'):
        env.reset()
    assert [c for c in calls if c[0]=='write']==[('write',(21,7,1)),('write',(21,7,0))]
    assert calls[-2:]==[('erase',7),('close',)]
    assert not env.last_reset_info['success']
