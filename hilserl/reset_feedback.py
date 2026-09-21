"""One reset-completion cue. No robot imports, device grabs or input events."""
import sys
import time

DURATION_MS = 800
STRONG_MAGNITUDE = 0xD000
WEAK_MAGNITUDE = 0xB000


def _open_controller(evdev):
    candidates = []
    try:
        for path in sorted(evdev.list_devices()):
            try:
                device = evdev.InputDevice(path)
            except OSError:
                continue
            keep = False
            try:
                keep = (device.info.vendor == 0x045E
                        and (device.info.product == 0x0B12 or "xbox" in device.name.lower())
                        and evdev.ecodes.FF_RUMBLE in
                        device.capabilities().get(evdev.ecodes.EV_FF, []))
                if keep:
                    candidates.append(device)
            finally:
                if not keep:
                    device.close()
        if len(candidates) != 1:
            raise RuntimeError(f"复位提醒需要唯一可震动 Xbox 手柄，当前找到 {len(candidates)} 个；请检查连接和权限。")
        return candidates.pop()
    finally:
        for device in candidates:
            device.close()


def rumble_once(*, check):
    """Block for one 800 ms pulse; cancel promptly and always release the effect.

    Hardware errors propagate so an episode cannot silently start without the
    requested ready cue. The caller keeps reset status throughout this function.
    """
    import evdev  # Optional Linux dependency, loaded only for an enabled cue.

    check()
    device = _open_controller(evdev)
    effect_id = None
    try:
        check()
        effect = evdev.ff.Effect(
            evdev.ecodes.FF_RUMBLE, -1, 0,
            evdev.ff.Trigger(0, 0), evdev.ff.Replay(DURATION_MS, 0),
            evdev.ff.EffectType(ff_rumble_effect=evdev.ff.Rumble(
                strong_magnitude=STRONG_MAGNITUDE, weak_magnitude=WEAK_MAGNITUDE)))
        effect_id = device.upload_effect(effect)
        check()
        started = time.monotonic()
        device.write(evdev.ecodes.EV_FF, effect_id, 1)
        while True:
            check()
            remaining = DURATION_MS / 1000 - (time.monotonic() - started)
            if remaining <= 0:
                break
            time.sleep(min(.05, remaining))
        return dict(state="played", duration_ms=DURATION_MS, pulse_count=1,
                    strong_magnitude=STRONG_MAGNITUDE, weak_magnitude=WEAK_MAGNITUDE,
                    device=device.path, device_name=device.name,
                    elapsed_seconds=time.monotonic() - started)
    finally:
        # Preserve cancellation/the primary error if unplugging also breaks cleanup.
        active_error = sys.exc_info()[1]
        cleanup_errors = []
        operations = []
        if effect_id is not None:
            operations += [lambda: device.write(evdev.ecodes.EV_FF, effect_id, 0),
                           lambda: device.erase_effect(effect_id)]
        operations.append(device.close)
        for operation in operations:
            try:
                operation()
            except OSError as exc:
                cleanup_errors.append(exc)
        if cleanup_errors and active_error is None:
            raise RuntimeError(f"手柄震动清理失败：{cleanup_errors}") from cleanup_errors[0]
