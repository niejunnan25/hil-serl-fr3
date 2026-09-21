"""Offline rumble and reset lifecycle tests; never open real input devices."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hilserl import reset_feedback as feedback
from test_plug_insertion_reset_safety import _Clock, _make_env


@pytest.fixture
def hardware(monkeypatch):
    clock = _Clock()
    calls = []
    device = SimpleNamespace(path='/dev/input/test', name='Microsoft Xbox Controller',
                             info=SimpleNamespace(vendor=0x045E, product=0x0B12))
    device.capabilities = lambda: {21: [80]}
    device.upload_effect = lambda effect: calls.append(('upload', effect)) or 7
    device.write = lambda *args: calls.append(('write', args))
    device.erase_effect = lambda ident: calls.append(('erase', ident))
    device.close = lambda: calls.append(('close',))
    fake = SimpleNamespace(list_devices=lambda: [device.path], InputDevice=lambda _: device,
        ecodes=SimpleNamespace(EV_FF=21, FF_RUMBLE=80),
        ff=SimpleNamespace(Effect=lambda *args: args, Trigger=lambda *args: args,
                           Replay=lambda *args: args, EffectType=lambda **kw: kw,
                           Rumble=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, 'evdev', fake)
    monkeypatch.setattr(feedback, 'time', clock)
    return clock, calls, device, fake


def test_single_800ms_pulse_approved_strength_and_cleanup(hardware):
    clock, calls, _, _ = hardware
    checked = []
    receipt = feedback.rumble_once(check=lambda: checked.append(clock.now))
    effect = calls[0][1]
    assert effect[4] == (800, 0)
    assert effect[5]['ff_rumble_effect'] == dict(strong_magnitude=0xD000, weak_magnitude=0xB000)
    assert [c for c in calls if c[0] == 'write'] == [('write', (21, 7, 1)), ('write', (21, 7, 0))]
    assert calls[-2:] == [('erase', 7), ('close',)]
    assert clock.now == pytest.approx(.8)
    assert max(np.diff(checked)) <= .050001
    assert receipt['pulse_count'] == 1 and receipt['duration_ms'] == 800


def test_stop_mid_pulse_releases_effect_without_finishing(hardware):
    clock, calls, _, _ = hardware
    class StopRequested(Exception):
        pass
    def check():
        if clock.now >= .3:
            raise StopRequested()
    with pytest.raises(StopRequested):
        feedback.rumble_once(check=check)
    assert .3 <= clock.now < .36
    assert calls[-3:] == [('write', (21, 7, 0)), ('erase', 7), ('close',)]


def test_cleanup_failure_does_not_mask_stop_or_skip_close(hardware):
    clock, calls, device, _ = hardware
    class StopRequested(Exception):
        pass
    original = device.write
    def write(*args):
        if args[-1] == 0:
            raise OSError('unplugged')
        original(*args)
    device.write = write
    def check():
        if clock.now > .1:
            raise StopRequested()
    with pytest.raises(StopRequested):
        feedback.rumble_once(check=check)
    assert calls[-2:] == [('erase', 7), ('close',)]


@pytest.mark.parametrize('case', ['none', 'unrelated', 'no_rumble', 'multiple'])
def test_device_discovery_never_pulses_unavailable_or_ambiguous_controller(hardware, case):
    _, calls, device, fake = hardware
    if case == 'none':
        fake.list_devices = lambda: []
    elif case == 'unrelated':
        device.info.vendor = 0x1234
    elif case == 'no_rumble':
        device.capabilities = lambda: {}
    else:
        fake.list_devices = lambda: ['a', 'b']
    with pytest.raises(RuntimeError, match='唯一'):
        feedback.rumble_once(check=lambda: None)
    assert not any(c[0] in {'upload', 'write'} for c in calls)


def test_reset_cue_follows_both_stable_stages_and_precedes_fresh_observation(monkeypatch):
    clock = _Clock()
    env, _ = _make_env(clock, moving=True)
    env.config.RESET_CLEAR_Z = .15
    env.config.RESET_FEEDBACK = True
    cue_times, observed = [], []
    def cue(*, check):
        assert env.last_reset_info['success'] is False
        assert env.last_reset_info['phase'] == 'notifying'
        motions = [p for p in env.last_reset_info['phases'] if p['name'] in ('clear','reset_pose')]
        assert len(motions) == 2 and all(p['state'] == 'converged' for p in motions)
        assert all(p['stable_seconds'] >= .5 for p in motions)
        assert all(phase == 'resetting' for phase, _ in env.published)
        cue_times.append(clock.now)
        clock.sleep(.8)
        check()
        return dict(state='played', duration_ms=800, pulse_count=1)
    monkeypatch.setattr(feedback, 'rumble_once', cue)
    env._get_obs = lambda: observed.append(clock.now) or {'time': clock.now}
    obs, info = env.reset()
    assert len(cue_times) == 1 and observed == [cue_times[0]+.8]
    assert obs['time'] == observed[0]
    assert info['reset']['success'] and info['reset']['feedback']['pulse_count'] == 1
    commands = [p for p in env.calls if isinstance(p,np.ndarray)]
    assert max(p[2] for p in commands) <= .15


def test_failed_reset_does_not_vibrate(monkeypatch):
    env, _ = _make_env(_Clock(), moving=False)
    env.config.RESET_FEEDBACK = True
    monkeypatch.setattr(feedback, 'rumble_once', lambda **_: pytest.fail('no cue after failed reset'))
    with pytest.raises(RuntimeError, match='stalled'):
        env.reset()


@pytest.mark.parametrize('stop', [False, True])
def test_notification_failure_or_stop_never_returns_ready_observation(monkeypatch, stop):
    env, _ = _make_env(_Clock(), moving=True)
    env.config.RESET_FEEDBACK = True
    class StopRequested(Exception):
        pass
    error = StopRequested if stop else RuntimeError
    def fail(**_):
        raise error('cue interrupted')
    monkeypatch.setattr(feedback, 'rumble_once', fail)
    env._get_obs = lambda: pytest.fail('must not start with stale observation')
    with pytest.raises(error):
        env.reset()
    assert not env.last_reset_info['success']
    assert env.last_reset_info['state'] == ('cancelled' if stop else 'failed')
    assert not any(p['name']=='ready' for p in env.last_reset_info['phases'])


def test_five_mm_plateau_completes_after_short_stability_window():
    clock = _Clock()
    env, _ = _make_env(clock)
    goal = env.currpos.copy()
    goal[0] += .005
    result = env.interpolate_move(goal, timeout=8, name='reset_pose')
    assert result[0] == pytest.approx(.005)
    assert .5 <= clock.now < 1.0
    assert env.last_reset_info['phases'][-1]['completion_criterion'] == 'stable_target'


@pytest.mark.parametrize('case', ['moving', 'jitter', 'rotation'])
def test_near_target_is_not_ready_while_moving_or_oscillating(case):
    from scipy.spatial.transform import Rotation
    clock = _Clock()
    env, _ = _make_env(clock)
    goal = env.currpos.copy()
    def update():
        if clock.now < 1.2:
            if case == 'moving':
                env.dq[:] = .04
            elif case == 'jitter':
                env.currpos[0] = .002 if round(clock.now*10)%2 else -.002
            else:
                env.currpos[3:] = Rotation.from_rotvec([.015 if round(clock.now*10)%2 else -.015,0,0]).as_quat()
        else:
            env.currpos = goal.copy()
            env.dq[:] = 0
    env._update_currpos = update
    env.interpolate_move(goal, timeout=8, name='reset_pose')
    assert 1.7 <= clock.now < 2.1


def test_clear_target_never_moves_down_to_withdraw():
    env, ns = _make_env(_Clock())
    env.config.RESET_CLEAR_Z = .15
    for z, expected in [(.09, .15), (.18, .18)]:
        env.currpos[2] = z
        result = ns['compute_reset_clear_pose'](env.currpos, env.config)
        assert result[2] == expected
        np.testing.assert_array_equal(result[[0,1,3,4,5,6]], env.currpos[[0,1,3,4,5,6]])
