#!/usr/bin/env bash
# Former encoded launcher: the desktop now opens the shared workbench.
set -euo pipefail
HILSERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HILSERL_ROOT/bin/hil-serl" console --background --open "$@"
