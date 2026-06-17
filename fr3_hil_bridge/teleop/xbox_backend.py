from __future__ import annotations

import os
import time
from typing import Any


def _base_payload(status: str, backend: str) -> dict[str, Any]:
    return {
        "ok": True,
        "status": status,
        "backend": backend,
        "runtime_execution_performed": False,
        "motion_command": False,
        "gripper_command": False,
        "droid_mutation": False,
        "openpi_dependency": False,
    }


def mock_probe(duration_sec: float) -> dict[str, Any]:
    payload = _base_payload("PLUG_ZED_XBOX_PROBE_MOCK_NO_MOTION", "mock")
    payload.update(
        {
            "duration_sec": duration_sec,
            "controller_device_opened": False,
            "pygame_imported": False,
            "sdl2_controller_imported": False,
            "devices": [],
        }
    )
    return payload


def _import_pygame():
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    return pygame


def _joystick_payload(joy) -> dict[str, Any]:
    return {
        "name": joy.get_name(),
        "guid": joy.get_guid(),
        "instance_id": joy.get_instance_id(),
        "axes": joy.get_numaxes(),
        "buttons": joy.get_numbuttons(),
        "hats": joy.get_numhats(),
    }


def pygame_joystick_probe(duration_sec: float) -> dict[str, Any]:
    pygame = _import_pygame()
    pygame.init()
    pygame.joystick.init()
    devices: dict[int, dict[str, Any]] = {}

    # Enumerate already-connected devices first; hotplug events alone miss the
    # common case where the controller was connected before this process starts.
    for index in range(pygame.joystick.get_count()):
        joy = pygame.joystick.Joystick(index)
        devices[joy.get_instance_id()] = _joystick_payload(joy)

    deadline = time.monotonic() + duration_sec
    samples = 0
    while time.monotonic() < deadline:
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED:
                joy = pygame.joystick.Joystick(event.device_index)
                devices[joy.get_instance_id()] = _joystick_payload(joy)
            elif event.type == pygame.JOYDEVICEREMOVED:
                devices.pop(event.instance_id, None)
        samples += 1
        time.sleep(0.01)

    payload = _base_payload("PLUG_ZED_XBOX_PROBE_PYGAME_NO_MOTION", "pygame")
    payload.update(
        {
            "pygame_imported": True,
            "sdl2_controller_imported": False,
            "controller_device_opened": bool(devices),
            "devices": list(devices.values()),
            "samples": samples,
        }
    )
    return payload


def _controller_payload(pygame, controller_module, joystick_index: int) -> dict[str, Any]:
    joy = pygame.joystick.Joystick(joystick_index)
    base = _joystick_payload(joy)
    base["joystick_index"] = joystick_index
    try:
        ctrl = controller_module.Controller.from_joystick(joy)
        base["sdl_controller"] = True
        base["attached"] = bool(ctrl.attached())
        base["controller_init"] = bool(ctrl.get_init())
        ctrl.quit()
    except Exception as exc:  # noqa: BLE001 - probe must return diagnostics
        base["sdl_controller"] = False
        base["sdl_controller_error"] = repr(exc)
    return base


def pygame_sdl_controller_probe(duration_sec: float) -> dict[str, Any]:
    pygame = _import_pygame()
    import pygame._sdl2.controller as controller_module

    pygame.init()
    pygame.joystick.init()
    devices: dict[int, dict[str, Any]] = {}

    for index in range(pygame.joystick.get_count()):
        item = _controller_payload(pygame, controller_module, index)
        devices[item["instance_id"]] = item

    deadline = time.monotonic() + duration_sec
    samples = 0
    while time.monotonic() < deadline:
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED:
                item = _controller_payload(pygame, controller_module, event.device_index)
                devices[item["instance_id"]] = item
            elif event.type == pygame.JOYDEVICEREMOVED:
                devices.pop(event.instance_id, None)
        samples += 1
        time.sleep(0.01)

    controller_devices = [item for item in devices.values() if item.get("sdl_controller") is True]
    payload = _base_payload("PLUG_ZED_XBOX_PROBE_SDL_CONTROLLER_NO_MOTION", "sdl")
    payload.update(
        {
            "pygame_imported": True,
            "sdl2_controller_imported": True,
            "joystick_device_opened": bool(devices),
            "controller_device_opened": bool(controller_devices),
            "devices": list(devices.values()),
            "controller_devices": controller_devices,
            "samples": samples,
        }
    )
    return payload
