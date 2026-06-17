#!/usr/bin/env bash
# =============================================================================
# 03_start_franka_server.sh — Start/stop/status franka_server.py
#
# franka_server.py is the SERL robot infra Flask server that provides HTTP
# endpoints for robot state and control. It communicates with the FR3 via
# ROS/franka_ros controllers.
#
# Default port: 5000 (upstream SERL default, matches config.py / verify_safety.py)
# Override port:  set SERL_FRANKA_PORT=5017 (hilserl-fr3 reserved port)
#
# Target host: fr3-desktop-ts (Ubuntu 22.04)
# Robot IP:    172.16.0.2
# =============================================================================

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
SERL_FR3_ROOT="${SERL_FR3_ROOT:-/home/robot/serl_projects/hil-serl-fr3}"
FRANKA_SERVER_PY=""
PORT="${SERL_FRANKA_PORT:-5000}"
ROBOT_IP="${SERL_ROBOT_IP:-172.16.0.2}"
LOG_DIR="${SERL_FR3_ROOT}/artifacts/logs"
PID_FILE="${LOG_DIR}/franka_server.pid"
LOG_FILE="${LOG_DIR}/franka_server.log"
HEALTH_TIMEOUT=10

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err() { echo -e "${RED}[✗]${NC} $1"; }
info() { echo -e "${CYAN}[i]${NC} $1"; }

# ─── Locate franka_server.py ─────────────────────────────────────────────────
find_franka_server() {
    # Check multiple candidate locations
    for candidate in \
        "${SERL_FR3_ROOT}/upstream/hil-serl/serl_robot_infra/robot_servers/franka_server.py" \
        "${SERL_FR3_ROOT}/upstream/serl/serl_robot_infra/robot_servers/franka_server.py"; do
        if [ -f "$candidate" ]; then
            FRANKA_SERVER_PY="$candidate"
            return 0
        fi
    done
    return 1
}

# ─── Health check ────────────────────────────────────────────────────────────
health_check() {
    local url="http://127.0.0.1:${PORT}/getstate"
    info "Health check: GET ${url}"

    local response
    local http_code

    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --connect-timeout 3 --max-time "$HEALTH_TIMEOUT" \
        "$url" 2>/dev/null || echo "000")

    if [ "$http_code" = "200" ]; then
        response=$(curl -s --connect-timeout 3 --max-time "$HEALTH_TIMEOUT" \
            "$url" 2>/dev/null || echo "")
        log "Server is healthy (HTTP ${http_code})"
        if [ -n "$response" ]; then
            # Pretty-print a summary
            echo "$response" | python3 -m json.tool 2>/dev/null | head -20 || echo "$response" | head -5
        fi
        return 0
    else
        err "Server not responding (HTTP ${http_code})"
        return 1
    fi
}

# ─── Commands ────────────────────────────────────────────────────────────────
cmd_start() {
    echo "=============================================="
    echo " Starting franka_server.py"
    echo "=============================================="
    echo ""

    # Check if already running
    if [ -f "$PID_FILE" ]; then
        local old_pid
        old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            err "franka_server.py already running (PID ${old_pid})"
            info "Use '$0 stop' to stop it first, or '$0 status' to check"
            exit 1
        else
            info "Removing stale PID file"
            rm -f "$PID_FILE"
        fi
    fi

    # Pre-flight checks
    info "Pre-flight checks..."

    # Check port availability
    if ss -ltnp 2>/dev/null | awk '{print $4}' | grep -Eq "(:|\\])${PORT}$"; then
        err "Port ${PORT} is already in use"
        ss -ltnp 2>/dev/null | grep ":${PORT}" || true
        exit 1
    fi
    log "Port ${PORT} is available"

    # Check robot connectivity
    if ping -c 1 -W 2 "$ROBOT_IP" &>/dev/null; then
        log "Robot reachable at ${ROBOT_IP}"
    else
        warn "Cannot reach robot at ${ROBOT_IP} — server may fail to connect"
    fi

    # Find franka_server.py
    if ! find_franka_server; then
        err "franka_server.py not found"
        info "Expected at: ${SERL_FR3_ROOT}/upstream/hil-serl/serl_robot_infra/robot_servers/franka_server.py"
        exit 1
    fi
    log "Found franka_server.py at ${FRANKA_SERVER_PY}"

    # Check for ROS (required by franka_server.py)
    local ros_setup=""
    for candidate in /opt/ros/noetic/setup.bash; do
        if [ -f "$candidate" ]; then
            ros_setup="$candidate"
            break
        fi
    done

    if [ -z "$ros_setup" ]; then
        err "ROS Noetic not found — franka_server.py requires ROS"
        info "Options:"
        info "  A) Install ROS Noetic"
        info "  B) Use the direct libfranka bridge (non-ROS) instead"
        info "     See: fr3_hil_bridge/ in the hilserl-fr3 project"
        exit 1
    fi

    # Create log directory
    mkdir -p "$LOG_DIR"

    # Activate environment
    local activation="${SERL_FR3_ROOT}/env/activation.sh"
    if [ -f "$activation" ]; then
        info "Activating hilserl-fr3 environment..."
        # shellcheck disable=SC1090
        source "$activation"
    fi

    # Source ROS
    info "Sourcing ROS..."
    # shellcheck disable=SC1090
    source "$ros_setup"

    # Source catkin workspace if it exists
    local catkin_ws="${SERL_FR3_ROOT}/catkin_ws_franka"
    if [ -f "${catkin_ws}/devel/setup.bash" ]; then
        info "Sourcing catkin workspace..."
        # shellcheck disable=SC1090
        source "${catkin_ws}/devel/setup.bash"
    fi

    # Start the server
    info "Starting franka_server.py on port ${PORT}..."
    info "Robot IP: ${ROBOT_IP}"
    info "Log file: ${LOG_FILE}"

    nohup python3 "$FRANKA_SERVER_PY" \
        --port "$PORT" \
        --robot-ip "$ROBOT_IP" \
        > "$LOG_FILE" 2>&1 &

    local server_pid=$!
    echo "$server_pid" > "$PID_FILE"

    # Wait for startup
    info "Waiting for server to start (PID ${server_pid})..."
    sleep 3

    if kill -0 "$server_pid" 2>/dev/null; then
        log "franka_server.py started (PID ${server_pid})"

        # Health check
        echo ""
        if health_check; then
            log "Server is ready!"
        else
            warn "Server started but health check failed"
            warn "Check logs: tail -f ${LOG_FILE}"
        fi
    else
        err "franka_server.py failed to start"
        err "Last 20 lines of log:"
        tail -20 "$LOG_FILE" 2>/dev/null || echo "(no log file)"
        rm -f "$PID_FILE"
        exit 1
    fi

    echo ""
    echo "=============================================="
    echo " Server Info"
    echo "=============================================="
    echo "  URL:     http://127.0.0.1:${PORT}"
    echo "  PID:     ${server_pid}"
    echo "  Log:     ${LOG_FILE}"
    echo "  Robot:   ${ROBOT_IP}"
    echo ""
    echo "  Endpoints:"
    echo "    GET  /getstate  — robot state"
    echo "    POST /pose      — send pose command"
    echo "    POST /reset     — reset robot"
    echo ""
    echo "  Stop with:  $0 stop"
    echo "  Logs with:  $0 logs"
    echo "=============================================="
}

cmd_stop() {
    echo "=============================================="
    echo " Stopping franka_server.py"
    echo "=============================================="
    echo ""

    if [ ! -f "$PID_FILE" ]; then
        warn "No PID file found at ${PID_FILE}"
        info "Trying to find franka_server.py processes..."

        local pids
        pids=$(pgrep -f "[f]ranka_server.py" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            info "Found franka_server.py processes: ${pids}"
            echo "$pids" | while read -r pid; do
                info "Sending SIGTERM to PID ${pid}..."
                kill "$pid" 2>/dev/null || true
            done
            sleep 2

            # Force kill if still running
            pids=$(pgrep -f "[f]ranka_server.py" 2>/dev/null || true)
            if [ -n "$pids" ]; then
                warn "Force killing remaining processes..."
                echo "$pids" | while read -r pid; do
                    kill -9 "$pid" 2>/dev/null || true
                done
            fi
            log "All franka_server.py processes stopped"
        else
            log "No franka_server.py processes found"
        fi
        return 0
    fi

    local pid
    pid=$(cat "$PID_FILE")

    if kill -0 "$pid" 2>/dev/null; then
        info "Sending SIGTERM to PID ${pid}..."
        kill "$pid" 2>/dev/null || true

        # Wait for graceful shutdown
        local waited=0
        while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 10 ]; do
            sleep 1
            waited=$((waited + 1))
        done

        if kill -0 "$pid" 2>/dev/null; then
            warn "Process did not stop gracefully, sending SIGKILL..."
            kill -9 "$pid" 2>/dev/null || true
        fi

        log "franka_server.py stopped"
    else
        info "Process ${pid} is not running"
    fi

    rm -f "$PID_FILE"
}

cmd_status() {
    echo "=============================================="
    echo " franka_server.py Status"
    echo "=============================================="
    echo ""

    # Check PID file
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log "Running (PID ${pid})"
        else
            warn "PID file exists but process ${pid} is dead"
        fi
    else
        info "No PID file"
    fi

    # Check processes
    local pids
    pids=$(pgrep -f "[f]ranka_server.py" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        info "Active processes: ${pids}"
    else
        info "No franka_server.py processes found"
    fi

    # Check port
    if ss -ltnp 2>/dev/null | awk '{print $4}' | grep -Eq "(:|\\])${PORT}$"; then
        log "Port ${PORT} is listening"
    else
        info "Port ${PORT} is not listening"
    fi

    # Health check
    echo ""
    health_check || true

    # Check robot
    echo ""
    if ping -c 1 -W 2 "$ROBOT_IP" &>/dev/null; then
        log "Robot reachable at ${ROBOT_IP}"
    else
        warn "Robot not reachable at ${ROBOT_IP}"
    fi
}

cmd_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        err "No log file at ${LOG_FILE}"
        exit 1
    fi
}

cmd_health() {
    health_check
}

# ─── Main ────────────────────────────────────────────────────────────────────
case "${1:-help}" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_stop
        sleep 2
        cmd_start
        ;;
    status)
        cmd_status
        ;;
    logs)
        cmd_logs
        ;;
    health)
        cmd_health
        ;;
    help|*)
        echo "Usage: $0 {start|stop|restart|status|logs|health}"
        echo ""
        echo "  start    — Start franka_server.py on port ${PORT}"
        echo "  stop     — Stop franka_server.py"
        echo "  restart  — Stop then start"
        echo "  status   — Check if server is running"
        echo "  logs     — Tail the server log"
        echo "  health   — Run health check (curl /getstate)"
        echo ""
        echo "Environment variables:"
        echo "  SERL_FRANKA_PORT  — Server port (default: 5000)"
        echo "  SERL_ROBOT_IP     — Robot IP (default: 172.16.0.2)"
        echo "  SERL_FR3_ROOT     — Project root"
        ;;
esac
