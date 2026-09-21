"""Exercise actual safety paths without importing hardware drivers."""
from types import SimpleNamespace
import numpy as np
import pytest
from test_plug_insertion_reset_safety import _Clock, _make_env
from hilserl.episodes import run_episodes


def state_for(env, mode=2):
    return dict(pose=env.currpos.tolist(), vel=[0.] * 6, force=[0., 0., 35.],
                torque=[0.] * 3, q=env.q.tolist(), dq=[0., .4938383694, 0., 0., 0., 0., 0.],
                jacobian=[0.] * 42, gripper_pos=.567, controller_running=True,
                robot_mode=mode, robot_mode_name='move' if mode == 2 else 'reflex',
                **{k+suffix: value for k in ('state', 'jacobian', 'gripper_state')
                   for suffix, value in (('_age_seconds', .01), ('_stale', False))})


def test_normal_recover_only_reads_state():
    env, ns = _make_env(_Clock())
    reads = []
    env._update_currpos = lambda: reads.append(True)
    ns['PlugInsertionEnv']._recover(env)
    assert reads == [True]
    assert env.calls == []


@pytest.mark.parametrize('mode', [0, 3, 4, 5, 6])
def test_fresh_protection_state_never_enters_motion(mode):
    env, ns = _make_env(_Clock())
    env._robot_post = lambda endpoint: SimpleNamespace(json=lambda: state_for(env, mode))
    with pytest.raises(RuntimeError, match='franka_safety_fatal'):
        ns['PlugInsertionEnv']._update_currpos(env)
    assert env._safety_rejected_state['robot_mode'] == mode
    assert not env.calls


@pytest.mark.parametrize('timeout', [False, True])
def test_stop_once_even_if_recorder_broken_and_timeout_unknown(monkeypatch, timeout):
    env, ns = _make_env(_Clock())
    env.url = 'http://no-robot.invalid/'
    env._command_pose = env.currpos.copy()
    env._safety_rejected_state = state_for(env, 4)
    env.recorder.check = lambda: (_ for _ in ()).throw(RuntimeError('broken recorder'))
    calls = []
    def post(url, **kwargs):
        calls.append(url)
        if timeout:
            raise ns['requests'].Timeout('lost reply')
        return SimpleNamespace(raise_for_status=lambda: None, status_code=200)
    monkeypatch.setattr(ns['requests'], 'post', post)
    monkeypatch.setattr(ns['requests'], 'get', lambda *a, **k: SimpleNamespace(
        json=lambda: {'controller_running': False, 'motion_available': False}))
    receipt = env.stop_for_safety('dq exceeded')
    assert env.stop_for_safety('second failure') is receipt
    assert calls == [env.url + 'stopimp']
    assert receipt['max_speed_joint'] == 2
    assert receipt['max_abs_dq'] == pytest.approx(.4938383694)
    assert receipt['sample']['robot_mode'] == 4
    assert receipt['sample']['command_pose'] == env.currpos.tolist()
    assert receipt['stop_confirmed'] is (not timeout)
    assert receipt['delivery_unknown'] is timeout
    assert env._command_pose is None
    with pytest.raises(RuntimeError, match='latched'):
        ns['PlugInsertionEnv']._robot_post(env, 'pose', json={'arr': env.currpos.tolist()})
    assert len(calls) == 1


@pytest.mark.parametrize('failure_at', ['before_step', 'step', 'reset'])
def test_episode_outer_boundary_stops_before_record_failure(failure_at):
    events = []
    def fail():
        raise RuntimeError('[actor_safety_fatal] dq=.49 > .35')
    base = SimpleNamespace(frame_references={}, _step_commands=[], raw_state=lambda: {'values': {}},
                           stop_for_safety=lambda reason: events.append('stop') or {'stop_confirmed': True})
    env = SimpleNamespace(unwrapped=base, close=lambda: None,
        reset=(fail if failure_at == 'reset' else lambda: ({}, {'reset': {'success': True}})),
        step=lambda action: fail())
    recorder = SimpleNamespace(check=lambda: None, event=lambda *a, **k: None,
        start_episode=lambda **k: 'ep', fail=lambda reason: events.append('record_failure'),
        close=lambda reason: None, error=None, episode=None)
    operator = SimpleNamespace(wait=lambda *a, **k: None, publish=lambda *a, **k: None,
        raise_if_stop=lambda: None, poll=lambda *a: None, state={'phase': 'fault'})
    with pytest.raises(RuntimeError, match='actor_safety_fatal'):
        run_episodes(env, lambda *a: (np.zeros(7), {}), recorder, operator,
                     before_step=lambda *a: fail() if failure_at == 'before_step' else None)
    assert events == ['stop', 'record_failure']

@pytest.mark.parametrize('measure', ['dq', 'force'])
def test_actual_step_stops_at_existing_limit_before_sending_action(monkeypatch, measure):
    env, ns = _make_env(_Clock())
    env.url = 'http://no-robot.invalid/'
    state = state_for(env)
    if measure == 'force':
        state['dq'] = [0.] * 7
        state['force'] = [0., 0., 46.]
    env._robot_post = lambda endpoint: SimpleNamespace(json=lambda: state)
    env._update_currpos = ns['PlugInsertionEnv']._update_currpos.__get__(env)
    calls = []
    monkeypatch.setattr(ns['requests'], 'post', lambda url, **kw: calls.append(url) or SimpleNamespace(
        raise_for_status=lambda: None, status_code=200))
    monkeypatch.setattr(ns['requests'], 'get', lambda *a, **kw: SimpleNamespace(
        json=lambda: {'controller_running': False}))
    with pytest.raises(RuntimeError, match='franka_safety_fatal'):
        env.step(np.zeros(7))
    assert calls == [env.url + 'stopimp']
    assert env._safety_stop_receipt['sample'][measure] == state[measure]
    assert env._step_commands == []
