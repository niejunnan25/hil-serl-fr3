"""One episode lifecycle for online training, fixed-policy evaluation and demos.

The environment owns device I/O. This loop owns collection boundaries and labels.
Recording is independent of the training replay sink, which may derive extra samples.
"""
from __future__ import annotations

import copy
import time
import numpy as np

from hilserl.action_contract import LEGACY, resolve_action_contract
from hilserl.control import StopRequested
from hilserl.errors import RobotStateUnavailable
from hilserl.storage import RecordingError, stamp


def _training_transition(raw, *, terminal=False, outcome=None, action_contract=LEGACY):
    contract = resolve_action_contract(action_contract)
    if raw.get("action_contract", contract.name) != contract.name:
        raise ValueError("raw and replay action contracts differ")
    info = raw["info"]
    actions = contract.to_learning_action(raw["actions"])
    if contract.fixed_xyz and "learning_action" in raw:
        recorded = contract.to_device_action(raw["learning_action"])
        if not np.array_equal(recorded, raw["actions"]):
            raise ValueError("recorded learning action differs from selected action")
    if terminal and contract.fixed_xyz and outcome not in (0, 1):
        raise ValueError("fixed-xyz-v1 terminal reward requires a human 0/1 verdict")
    transition = dict(
        observations=raw["observations"], actions=actions,
        next_observations=raw["next_observations"],
        rewards=float(outcome) if terminal else 0.0 if contract.fixed_xyz else raw["observed_reward"],
        dones=bool(terminal), masks=0.0 if terminal else 1.0,
        infos=dict(source_action=raw["source_action"], manual_success=bool(terminal),
                   succeed=bool(outcome) if terminal else False, step=raw["global_step"],
                   raw_transition_id=raw["id"], episode_id=raw["episode_id"],
                   termination_reason=raw.get("termination_reason"),
                   verdict_source="human" if terminal else None,
                   action_contract=contract.name),
    )
    transition["infos"]["image_profile"] = raw.get("image_profile", "full-frame128-v1")
    if not contract.fixed_xyz and "grasp_penalty" in info:
        transition["grasp_penalty"] = info["grasp_penalty"]
    return transition


def run_episodes(env, sample_action, recorder, operator, *, mode="train", max_episode_steps=190,
                 max_total_steps=100000, max_episodes=None, emit=lambda transition: None,
                 on_episode=lambda summary: None, before_step=lambda obs, step: None,
                 on_exit=lambda: None, action_contract=LEGACY, image_profile="full-frame128-v1", eval_seed=None,
                 episode_reward=None, refresh_observation=None):
    """No device step occurs in waiting/reset-label phases.

One replay transition is held until the next decision boundary, so a label arriving
between ticks can terminate the last real step without sending an additional action.
"""
    contract = resolve_action_contract(action_contract)
    if hasattr(env, "action_space") and env.action_space.shape != (contract.learning_dim,):
        raise ValueError(f"{contract.name} requires environment action shape ({contract.learning_dim},)")
    base = env.unwrapped
    global_step = 0
    completed = []
    reason = "stopped"
    recovery_error = None

    def emit_replay(transition):
        if episode_reward is None:
            emit(transition)

    def check_recording():
        recorder.check()
        if episode_reward is not None:
            episode_reward.check()
            operator.publish(reward=episode_reward.snapshot())

    try:
        if episode_reward is not None:
            if not callable(refresh_observation):
                raise ValueError("Episode reward mode requires an action-free observation refresh")
            episode_reward.recover(operator, recorder)
        while global_step < max_total_steps and (max_episodes is None or len(completed) < max_episodes):
            pending = None
            phase = "waiting_reset"
            try:
                operator.publish("waiting_reset", episode_id=None, step=0, completed_episodes=len(completed))
                recorder.event("phase", phase="waiting_reset")
                operator.wait("waiting_reset",
                              ("状态反馈中断，Actor 已暂停；已有步骤保留。"
                               "恢复底层状态并检查机器人后，点击复位开始新的 episode；不会续发旧动作。"
                               if recovery_error else "处理好插头和线缆，确认 E-stop 在手后，按 Enter 或点击复位开始。"),
                              {"continue"}, check=check_recording)
                # Xbox readiness is established without taking dummy environment steps.
                wrapper = env
                while wrapper is not None:
                    method = vars(type(wrapper)).get("wait_for_ready")
                    if method:
                        method(wrapper, operator, recorder)
                    wrapper = vars(wrapper).get("env")
                recorder.check()
                operator.publish("resetting", prompt="正在复位")
                recorder.event("phase", phase="resetting")
                phase = "resetting"
                base._step_commands = []
                if mode == "eval" and eval_seed is not None:
                    # The next evaluation case is stable even after a failed
                    # reset or an incomplete recovery episode is retried.
                    np.random.seed((int(eval_seed) + len(completed)) % 2**32)
                obs, reset_info = env.reset()
                recorder.check()
                # `succeed` is the task reward flag, not evidence of reset success.
                # A returned observation alone must never open a collection window.
                reset = reset_info.get("reset") if isinstance(reset_info, dict) else None
                if not isinstance(reset, dict) or reset.get("success") is not True:
                    recorder.event("reset_rejected", reset_info=_plain_info(reset_info))
                    raise RuntimeError("复位未通过到位验证，禁止开始采集；请查看复位诊断。")
                operator.raise_if_stop()
                recorder.event("reset_verified", reset=_plain_info(reset))
                if episode_reward is not None:
                    episode_reward.barrier(operator, recorder)
                    operator.raise_if_stop()
                    obs = refresh_observation()
                    check_recording()
                    operator.raise_if_stop()
                    recorder.event("observation_refreshed", after_reward_barrier=True)
                episode_id = recorder.start_episode(mode=mode, reset_info=_plain_info(reset_info),
                                                    action_contract=contract.name, image_profile=image_profile)
                if episode_reward is not None:
                    episode_reward.begin(episode_id)
                recovery_error = None
                phase = "collecting"
                episode_start = stamp()
                operator.publish("collecting", episode_id=episode_id, step=0, error=None,
                                 collection_started_monotonic_ns=episode_start["monotonic_ns"],
                                 robot_state_error=None, recoverable=False, prompt="0 失败结束 · 1 成功结束")
                pending = None
                deferred_command = None
                episode_steps, human_steps = 0, 0
                while True:
                    check_recording()
                    command = deferred_command or operator.poll({"label", "pause"})
                    deferred_command = None
                    if command and command["command"] == "pause":
                        if pending is not None:
                            emit_replay(_training_transition(pending, action_contract=contract))
                        recorder.finish_episode(complete=False, reason="operator_pause")
                        if episode_reward is not None:
                            episode_reward.abort("operator_pause")
                        operator.wait("paused", "Actor 已暂停；处理完成后继续，将开始新的 episode。", {"continue"}, check=recorder.check)
                        break
                    terminal_reason = None
                    if command and command["command"] == "label":
                        terminal_reason = "manual"
                    elif pending is not None:
                        terminal_reason = pending.get("termination_reason")
                        if not terminal_reason and episode_steps >= max_episode_steps:
                            terminal_reason = "time_limit"
                        if not terminal_reason and global_step >= max_total_steps:
                            terminal_reason = "run_step_limit"
                    if terminal_reason:
                        collection_ended = pending["step_ended"] if pending else episode_start
                        if episode_reward is not None:
                            operator.publish("awaiting_label" if command is None or command["command"] != "label"
                                             else "reward_pending")
                            episode_reward.end_collection()
                        recorder.end_collection(terminal_reason, ended=collection_ended)
                        if command is None or command["command"] != "label":
                            command = operator.wait("awaiting_label", "本条采集已结束，请输入 0=失败 或 1=成功。",
                                                    {"label"}, check=check_recording)
                        outcome = int(command["value"])
                        if recorder.episode:
                            recorder.episode.update(verdict_source="human", verdict_time=stamp(),
                                                    human_steps=human_steps)
                        finalized = recorder.finish_episode(outcome)
                        if not finalized["complete"]:
                            operator.publish("waiting_reset", prompt="本条没有有效执行步，未计入完成数量。")
                            break
                        if pending is not None:
                            pending["termination_reason"] = terminal_reason
                            emit_replay(_training_transition(pending, terminal=True, outcome=outcome, action_contract=contract))
                        if episode_reward is not None:
                            episode_reward.finalize(outcome, terminal_reason)
                        summary = dict(episode_id=episode_id, outcome=outcome, steps=episode_steps,
                                       human_steps=human_steps, mode=mode, termination_reason=terminal_reason,
                                       collection_seconds=(collection_ended["monotonic_ns"] - episode_start["monotonic_ns"]) / 1e9)
                        completed.append(summary)
                        on_episode(summary)
                        operator.publish("waiting_reset", latest_episode=summary, completed_episodes=len(completed))
                        break
                    before_step(obs, global_step)
                    raw_before = base.raw_state()
                    frames_before = copy.deepcopy(base.frame_references)
                    sample_started = stamp()
                    proposed_learning, policy_metadata = sample_action(obs, global_step)
                    proposed = contract.to_device_action(proposed_learning)
                    policy_action = contract.project_device_action(proposed)
                    environment_action = contract.to_learning_action(policy_action)
                    recorder.check()
                    deferred_command = operator.poll({"label", "pause"})
                    if deferred_command:
                        continue
                    if pending is not None:
                        emit_replay(_training_transition(pending, action_contract=contract))
                        pending = None
                    step_started = stamp()
                    base._step_commands = []
                    base._action_contract_attempt = None
                    try:
                        next_obs, reward, terminated, truncated, info = env.step(environment_action)
                    except BaseException as exc:
                        commands = copy.deepcopy(getattr(base, "_step_commands", []))
                        if commands:
                            # Outer observation/reward wrappers can fail after sending a valid pose.
                            # Retain the attempt without inventing an unavailable next observation.
                            recorder.append_step(dict(
                                id=f"{recorder.directory.parents[1].name}/{recorder.directory.name}/{episode_id}/{episode_steps:06d}",
                                episode_id=episode_id, episode_step=episode_steps, global_step=global_step,
                                complete_transition=False, error=type(exc).__name__, observations=copy.deepcopy(obs),
                                next_observations=None, proposed_action=proposed, policy_action=policy_action,
                                action_contract=contract.name, image_profile=image_profile, learning_action=None,
                                action_attempt=copy.deepcopy(base._action_contract_attempt),
                                actions=None, source_action=None, policy=policy_metadata, raw_state=raw_before,
                                raw_next_state=None, frame_references=frames_before,
                                error_details=exc.as_dict() if isinstance(exc, RobotStateUnavailable) else str(exc),
                                next_frame_references=copy.deepcopy(base.frame_references), controller_commands=commands,
                                step_started=step_started, step_ended=stamp()), recovery=True)
                        recorder.event("step_failed", episode_id=episode_id, episode_step=episode_steps,
                                       error=type(exc).__name__, command_count=len(commands))
                        raise
                    step_ended = stamp()
                    source = "human" if mode == "collect" or "intervene_action" in info else "policy"
                    chosen = contract.project_device_action(contract.to_device_action(
                        info.get("intervene_action", environment_action)))
                    learning_action = contract.to_learning_action(chosen)
                    human_steps += int(source == "human")
                    terminal_reason = info.get("termination_reason")
                    if terminated and not terminal_reason:
                        terminal_reason = "environment_terminate"
                    if truncated and not terminal_reason:
                        terminal_reason = "time_limit"
                    pending = dict(
                        id=f"{recorder.directory.parents[1].name}/{recorder.directory.name}/{episode_id}/{episode_steps:06d}",
                        episode_id=episode_id, complete_transition=True,
                        episode_step=episode_steps, global_step=global_step, observations=copy.deepcopy(obs),
                        next_observations=copy.deepcopy(next_obs), proposed_action=proposed, policy_action=policy_action,
                        actions=chosen, source_action=source, policy=policy_metadata,
                        action_contract=contract.name, image_profile=image_profile, learning_action=learning_action,
                        requested_action=np.asarray(info.get("requested_intervene_action",
                                                            info.get("intervene_action", proposed)), dtype=np.float32).copy(),
                        requested_action_frame="base" if "requested_intervene_action" in info else "body",
                        requested_controller_action=info.get("requested_controller_action"),
                        raw_state=raw_before, raw_next_state=info.get("raw_next_state", base.raw_state()),
                        frame_references=frames_before, next_frame_references=copy.deepcopy(base.frame_references),
                        controller_input_action=info.get("controller_input_action"),
                        controller_commands=info.get("controller_commands", []),
                        observed_reward=float(reward), terminated=bool(terminated), truncated=bool(truncated),
                        termination_reason=terminal_reason, sample_started=sample_started,
                        step_started=step_started, step_ended=step_ended, info=_plain_info(info),
                    )
                    recorder.append_step(pending, recovery=True)
                    if episode_reward is not None:
                        episode_reward.append(_training_transition(pending, action_contract=contract), pending)
                    episode_steps += 1
                    global_step += 1
                    obs = next_obs
                    operator.publish("collecting", step=episode_steps, global_step=global_step,
                                     human_steps=human_steps, classifier=_plain_info(info.get("classifier", {})))
            except RobotStateUnavailable as exc:
                recovery_error = exc.as_dict()
                # Publish before draining the recorder so the Learner can pause
                # promptly. A new gate is issued only by the next operator.wait.
                operator.publish("waiting_reset", episode_id=None, step=0, gate_id=None,
                                 error=str(exc), robot_state_error=recovery_error, recoverable=True,
                                 prompt="状态反馈中断，Actor 已暂停；恢复状态后请重新确认复位。")
                recorder.event("robot_state_unavailable", phase=phase,
                               episode_id=recorder.episode["id"] if recorder.episode else None,
                               error=recovery_error,
                               reset_info=_plain_info(getattr(exc, "reset_info", None)),
                               reset_commands=_plain_info(getattr(base, "_step_commands", []))
                               if phase == "resetting" else [])
                if recorder.episode:
                    recorder.episode.update(robot_state_error=recovery_error,
                                            termination_reason="robot_state_unavailable")
                    # The retained step has real observations, but no human
                    # verdict. Do not synthesize a terminal reward or success.
                    if pending is not None:
                        emit_replay(_training_transition(pending, action_contract=contract))
                    recorder.finish_episode(complete=False, reason="robot_state_unavailable")
                    if episode_reward is not None:
                        episode_reward.abort("robot_state_unavailable")
                recorder.check()  # finish_episode(complete=False) may retain a writer fault.
                operator.raise_if_stop()  # Read queued stop without executing gripper commands.
                if hasattr(operator, "pending"):
                    operator.pending[:] = [command for command in operator.pending
                                           if command.get("command") in ("stop", "abort", "quit", "exit")]
                pending = None
                # The outer loop waits for one *new* confirmation before reset.
                # No recover/home/reset/pose/gripper request is issued here.
        if episode_reward is not None:
            episode_reward.barrier(operator, recorder, final=True)
        reason = "completed"
    except (StopRequested, KeyboardInterrupt):
        reason = "operator_stop"
    except Exception as exc:
        reason = str(exc)
        recorder.fail(reason)
        operator.publish("fault", error=reason, prompt="Actor 已暂停。处理记录或设备问题后可重新启动；已有数据保留。")
        raise
    finally:
        # Stop camera producers before completing encoder queues. Never reset here.
        try:
            if episode_reward is not None:
                try:
                    if reason != "completed":
                        episode_reward.abort(reason)
                finally:
                    episode_reward.close()
            on_exit()
        except Exception as exc:
            reason = str(exc)
            recorder.fail(reason)
            operator.publish("fault", error=reason)
            raise
        finally:
            try:
                env.close()
            finally:
                recorder.close(reason)
                if recorder.error:
                    operator.publish("fault", error=recorder.error)
                elif operator.state["phase"] != "fault":
                    operator.publish("stopped", reason=reason, completed_episodes=len(completed))
    return completed


def _plain_info(info):
    """Small diagnostic fields; full arrays have named columns in the raw record."""
    omitted = {"raw_next_state", "controller_commands", "controller_input_action", "intervene_action", "original_state_obs",
               "requested_controller_action", "requested_intervene_action", "selected_action", "learning_action"}
    if isinstance(info, dict):
        return {str(k): _plain_info(v) for k, v in info.items() if k not in omitted}
    if isinstance(info, (np.ndarray, np.generic)):
        return info.tolist()
    if isinstance(info, (list, tuple)):
        return [_plain_info(x) for x in info]
    if info is None or isinstance(info, (str, int, float, bool)):
        return info
    return repr(info)
