#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Same as start_e2e.sh, but serves the UI on the machine's public IP so it can be
# opened from other computers. The browser only talks to the UI gateway
# (frontends/ui/server.js), which proxies every backend call server-side, so the
# backend stays bound to localhost and is not exposed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
UI_DIR="$PROJECT_ROOT/frontends/ui"
VENV_DIR="${DEEP_RESEARCHER_VENV_DIR:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
NAT_BIN="$VENV_DIR/bin/nat"

# Default config file
CONFIG_FILE="configs/config_web_default_llamaindex.yml"
PORT=8000
START_OPENSHELL_GATEWAY=false
UI_PORT=3000
PUBLIC_IP="${PUBLIC_IP:-}"
BACKEND_PID=""
FRONTEND_PID=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config_file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --start-openshell-gateway)
            START_OPENSHELL_GATEWAY=true
            shift
            ;;
        --public-ip)
            PUBLIC_IP="$2"
            shift 2
            ;;
        --ui-port)
            UI_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config_file <path>  Path to config file (default: configs/config_web_default_llamaindex.yml)"
            echo "  --port PORT           Backend server port (default: 8000)"
            echo "  --start-openshell-gateway  Start/verify an authenticated gateway and run its strict capability probe"
            echo "  --public-ip IP        Public IP to print and allow (default: \$PUBLIC_IP, else auto-detected)"
            echo "  --ui-port PORT        UI port, served on all interfaces (default: 3000)"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --config_file configs/config_web_default_llamaindex.yml --port 8000"
            echo "  $0 --start-openshell-gateway --config_file configs/config_openshell.yml --port 8000"
            echo "  $0 --public-ip 203.0.113.10 --ui-port 3000"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate config file exists
if [ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 --config_file <path>"
    echo "Example: $0 --config_file configs/config_web_default_llamaindex.yml"
    exit 1
fi

cleanup() {
    echo ""
    echo "Shutting down services..."
    if [[ -n "${BACKEND_PID:-}" ]]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [[ -n "${FRONTEND_PID:-}" ]]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

detect_public_ip() {
    if [[ -n "$PUBLIC_IP" ]]; then
        return
    fi
    # Ask an external service first: on cloud VMs behind NAT, the public IP is not
    # assigned to any local interface. Fall back to the first local address.
    PUBLIC_IP="$(curl -s -m 5 https://checkip.amazonaws.com 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ ! "$PUBLIC_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        PUBLIC_IP="$(hostname -I | awk '{print $1}')"
    fi
    if [[ -z "$PUBLIC_IP" ]]; then
        echo "Error: could not detect the public IP. Pass it with --public-ip."
        exit 1
    fi
}

echo "================================================"
echo "Starting Deep Researcher Agent Blueprint (End-to-End, public IP)"
echo "================================================"
echo ""

check_env() {
    export DEEP_RESEARCHER_DEV_ENV=e2e
    echo "Set DEEP_RESEARCHER_DEV_ENV=e2e"

    if [ -f "./deploy/.env" ]; then
        set -a  # Automatically export all variables
        source ./deploy/.env
        set +a  # Stop auto-exporting
        echo "Backend environment file loaded (deploy/.env)"
    else
        echo "No deploy/.env file found (optional)"
    fi

    # Suppress Python warnings unless overridden by .env
    export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

    # The UI gateway reaches the backend over localhost; browsers never call it directly.
    export BACKEND_URL="http://localhost:${PORT}"
    export NEXT_PUBLIC_BACKEND_URL="http://localhost:${PORT}"
    echo "Backend URL for e2e: $BACKEND_URL"

    detect_public_ip
    echo "Public IP: $PUBLIC_IP"
}

check_dependencies() {
    echo "Checking Python dependencies..."

    if [ ! -x "$PYTHON_BIN" ]; then
        echo "Deep Researcher Agent virtual environment not found at $VENV_DIR"
        echo "Run ./scripts/setup.sh or ./scripts/openshell/setup_openshell.sh first."
        exit 1
    fi

    if ! "$PYTHON_BIN" -c "import nat" 2>/dev/null; then
        echo "NAT not installed. Installing dependencies..."
        "$PYTHON_BIN" -m pip install -e .
    fi
    if [ ! -x "$NAT_BIN" ]; then
        echo "NAT CLI not found at $NAT_BIN"
        echo "Run ./scripts/setup.sh or ./scripts/openshell/setup_openshell.sh first."
        exit 1
    fi

    echo "Python dependencies installed ($PYTHON_BIN)"
}

check_openshell_component_versions() {
    if ! grep -Eq '^[[:space:]]*provider:[[:space:]]*openshell([[:space:]]|$)' "$PROJECT_ROOT/$CONFIG_FILE"; then
        return
    fi
    echo "Checking the certified OpenShell component stack..."
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/openshell/check_versions.py" \
        --gateway-name "${DEEP_RESEARCHER_OPENSHELL_GATEWAY_NAME:-openshell}"
}

check_ui_dependencies() {
    if [ ! -d "$UI_DIR" ]; then
        echo "UI directory not found at $UI_DIR"
        echo "Skipping frontend startup"
        return 1
    fi

    cd "$UI_DIR"

    if [ ! -d "node_modules" ]; then
        echo "Installing UI dependencies..."
        if command -v npm &> /dev/null; then
            npm ci
            echo "UI dependencies installed"
        else
            echo "npm not found. Skipping UI setup."
            echo "   Install Node.js 22+ to enable UI features"
            cd "$PROJECT_ROOT"
            return 1
        fi
    else
        echo "UI dependencies already installed"
    fi

    cd "$PROJECT_ROOT"
    return 0
}

start_backend() {
    echo ""
    echo "================================================"
    echo "Starting NAT Backend Server (Hot Reload Enabled)..."
    echo "================================================"
    echo ""
    echo "Backend will be available at: http://localhost:${PORT}"
    echo "Backend will auto-reload on code changes"
    echo "Config: $CONFIG_FILE"
    echo ""

    # Bound to localhost only: the UI proxies every backend call, so the API is not
    # exposed on the public IP.
    "$NAT_BIN" serve --config_file "$CONFIG_FILE" --host 127.0.0.1 --port "$PORT" &
    BACKEND_PID=$!
    echo "Backend PID: $BACKEND_PID"
}

start_openshell_gateway() {
    if [[ "$START_OPENSHELL_GATEWAY" != "true" ]]; then
        return
    fi
    echo "Starting/verifying authenticated OpenShell gateway..."
    "$PROJECT_ROOT/scripts/openshell/start_openshell_gateway.sh"
}

wait_for_backend() {
    echo "Waiting for backend to be ready..."
    local max_attempts=150
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:${PORT}/health" > /dev/null 2>&1 || \
           curl -s -f "http://localhost:${PORT}/docs" > /dev/null 2>&1; then
            echo "Backend is ready!"
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo ""
    echo "Backend health check timeout after ${max_attempts}s"
    echo "   Continuing anyway - frontend may encounter initial connection errors"
    return 1
}

start_frontend() {
    if [ ! -d "$UI_DIR" ]; then
        return
    fi

    echo ""
    echo "================================================"
    echo "Starting UI Frontend..."
    echo "================================================"
    echo ""
    echo "Frontend will be available at: http://${PUBLIC_IP}:${UI_PORT}"
    echo ""

    cd "$UI_DIR"

    # Same two processes as `npm run dev`, started directly so each binds where it should:
    # - server.js (the UI gateway) listens publicly on HOSTNAME:PORT. Both are set
    #   explicitly: bash defines HOSTNAME as the machine name, and PORT in this script
    #   is the backend port.
    # - The Next.js dev server stays on localhost; the gateway proxies to it.
    # NEXT_ALLOWED_DEV_ORIGINS lets the browser load dev resources (HMR) from the public
    # IP, which Next.js blocks by default (read by next.config.ts).
    HOSTNAME=0.0.0.0 PORT="$UI_PORT" \
        NEXT_INTERNAL_URL="http://127.0.0.1:3001" \
        NEXT_ALLOWED_DEV_ORIGINS="$PUBLIC_IP" \
        npx concurrently --kill-others --names ",next" \
        "node server.js" \
        "npx next dev --turbopack -p 3001 -H 127.0.0.1" &
    FRONTEND_PID=$!
    echo "Frontend PID: $FRONTEND_PID"

    cd "$PROJECT_ROOT"
}

main() {
    check_env
    echo ""

    check_dependencies
    echo ""

    start_openshell_gateway
    echo ""

    check_openshell_component_versions
    echo ""

    if check_ui_dependencies; then
        HAS_UI=true
    else
        HAS_UI=false
    fi
    echo ""

    start_backend
    wait_for_backend

    if [ "$HAS_UI" = true ]; then
        start_frontend
    else
        echo ""
        echo "WARNING: Frontend will NOT be started."
        echo "   Reason: UI dependencies not available (missing npm or node_modules)"
        echo "   To fix: install Node.js 22+ and run 'npm ci' in frontends/ui/"
        echo "   The backend will still run at http://localhost:${PORT}"
        echo ""
    fi

    echo ""
    echo "================================================"
    echo "Services Started"
    echo "================================================"
    echo ""
    echo "Backend: http://localhost:${PORT} (local only)"
    if [ "$HAS_UI" = true ]; then
        echo "Frontend: http://${PUBLIC_IP}:${UI_PORT}"
    else
        echo "Frontend: SKIPPED (see warning above)"
    fi
    echo ""
    echo "Make sure port ${UI_PORT} is open in the machine's firewall or cloud security group."
    if [[ -n "${LOGIN_USERNAME:-}" && -n "${LOGIN_PASSWORD:-}" ]]; then
        echo ""
        echo "Sign-in required: LOGIN_USERNAME/LOGIN_PASSWORD protect the UI (http://${PUBLIC_IP}:${UI_PORT}/login)."
    elif [[ "${REQUIRE_AUTH:-false}" != "true" ]]; then
        echo ""
        echo "WARNING: REQUIRE_AUTH is not enabled. Anyone who can reach"
        echo "   http://${PUBLIC_IP}:${UI_PORT} can use the app and your API keys."
    fi
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo ""

    wait
}

main
