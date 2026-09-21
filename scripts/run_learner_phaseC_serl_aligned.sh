#!/usr/bin/env bash
# Compatibility entry: configuration and process ownership live in hilserl/.
set -euo pipefail
HILSERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$HILSERL_ROOT/bin/hil-serl" train --learner-only "$@"
