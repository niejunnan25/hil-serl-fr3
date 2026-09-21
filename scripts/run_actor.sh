#!/usr/bin/env bash
# Deprecated legacy actor launcher.
#
# This file is intentionally fail-closed. The live Phase C actor must be started
# with bin/hil-serl train, which sets the local learner endpoint,
# rotation locks, reset/manual gates, and real-robot safety caps.

set -euo pipefail

cat >&2 <<'EOF'
[run_actor.sh refused]
This legacy launcher is disabled because it used the stale experiments actor,
old learner endpoint, and did not carry the Phase C safety environment.

Use:
  bash bin/hil-serl train

For eval-only use, call:
  bin/hil-serl eval --checkpoint <path>

RUN_ACTOR_LEGACY_ACK is not supported; this entrypoint must stay fail-closed.
EOF

exit 64
