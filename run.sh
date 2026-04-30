#!/usr/bin/env bash
# Launch the AI Content Detector Streamlit app.
# Activates the local venv and starts streamlit on http://localhost:8501.

set -euo pipefail

# Resolve the directory this script lives in, so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "Error: .venv not found in $SCRIPT_DIR" >&2
    echo "Build it once with:" >&2
    echo "  python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

exec streamlit run app.py "$@"
