#!/usr/bin/env bash
# Compatibility entry for the former two-role launcher.
set -euo pipefail
HILSERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--actor" ]]; then
    shift
    exec "$HILSERL_ROOT/bin/hil-serl" train "$@"
fi
exec "$HILSERL_ROOT/bin/hil-serl" train --learner-only "$@"
