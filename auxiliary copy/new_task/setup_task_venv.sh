#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./auxiliary/new_task/setup_task_venv.sh [-y|--yes] <task_name> <deps_file> [<examples_dir>]
# Examples:
#   ./auxiliary/new_task/setup_task_venv.sh circle_packing tasks/circle_packing/requirements.txt
#   ./auxiliary/new_task/setup_task_venv.sh --yes circle_packing tasks/circle_packing/pyproject.toml /abs/path/to/examples
#
# This script:
# - Creates a uv venv at <examples_dir>/<task_name>/venv  (default examples_dir = <repo_root>/examples)
# - Installs Shinka (editable, --no-deps) from repo root
# - Installs hydra-core==1.3.2
# - Installs dependencies from the provided file:
#     * requirements.txt: direct install with -r
#     * pyproject.toml: parses [project].dependencies (extras/groups ignored)
#
# It asks for confirmation before creating or modifying anything.
# Use -y/--yes to auto-confirm (non-interactive).

# Parse optional -y/--yes
AUTO_YES=0
if [[ $# -ge 1 ]]; then
  case "$1" in
    -y|--yes)
      AUTO_YES=1
      shift
      ;;
  esac
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 [-y|--yes] <task_name> <deps_file> [<examples_dir>]"
  exit 1
fi

TASK_NAME="$1"
DEPS_FILE="$2"

# Resolve repo root based on script location (works regardless of current working directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Optional 3rd argument: custom examples directory
DEFAULT_EXAMPLES_DIR="$REPO_ROOT/examples"
EXAMPLES_DIR="${3:-$DEFAULT_EXAMPLES_DIR}"

# Pre-flight checks (no changes performed here)
if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is not installed. Install it first (e.g., 'pip install uv')."
  exit 1
fi

if [[ ! -d "$REPO_ROOT/shinka" ]]; then
  echo "Error: Could not locate repo root (expected ./shinka under $REPO_ROOT)."
  echo "Resolved repo root: $REPO_ROOT"
  exit 1
fi

if [[ ! -f "$DEPS_FILE" ]]; then
  echo "Error: Dependencies file not found: $DEPS_FILE"
  exit 1
fi

TASK_DIR="$EXAMPLES_DIR/$TASK_NAME"
VENV_DIR="$TASK_DIR/venv"
PYTHON="$VENV_DIR/bin/python"

# Show the plan and ask for confirmation
echo "==> Plan"
echo "    Repo root:        $REPO_ROOT"
echo "    Examples dir:     $EXAMPLES_DIR"
echo "    Task name:        $TASK_NAME"
echo "    Task dir:         $TASK_DIR"
echo "    Venv dir:         $VENV_DIR"
echo "    Python executable:$PYTHON"
echo "    Deps file:        $DEPS_FILE"
echo "    Actions:"
echo "      - Create uv venv if missing"
echo "      - Install Shinka (editable, --no-deps) from repo root"
echo "      - Install hydra-core==1.3.2"
echo "      - Install dependencies from the provided file"

if [[ "$AUTO_YES" -eq 0 ]]; then
  read -r -p "Proceed with these actions? [y/N] " REPLY
  case "$REPLY" in
    y|Y) ;;
    *) echo "Aborted."; exit 0 ;;
  esac
else
  echo "(Auto-confirm enabled via -y/--yes)"
fi

# Perform actions
mkdir -p "$TASK_DIR"

echo "==> Creating uv venv at: $VENV_DIR"
if [[ -d "$VENV_DIR" ]]; then
  echo "    (venv already exists; reusing)"
else
  uv venv "$VENV_DIR"
fi

echo "==> Installing Shinka (editable, no dependencies) from $REPO_ROOT"
uv pip install -p "$PYTHON" --no-deps -e "$REPO_ROOT"

echo "==> Installing hydra-core==1.3.2"
uv pip install -p "$PYTHON" "hydra-core==1.3.2"

# install_from_pyproject() {
#   local toml_path="$1"
#   local tmp_req
#   tmp_req="$(mktemp /tmp/pyproj-reqs.XXXXXX.txt)"

#   "$PYTHON" - <<'PY' "$toml_path" >"$tmp_req" || {
# import sys, os
# py = sys.version_info
# USE_TOMLLIB = (py.major, py.minor) >= (3, 11)
# try:
#     if USE_TOMLLIB:
#         import tomllib as tomlmod
#     else:
#         import tomli as tomlmod
# except Exception:
#     sys.stderr.write("Could not import tomllib/tomli to parse pyproject.toml.\n"
#                      "Install 'tomli' into this env or use a requirements.txt.\n")
#     sys.exit(1)

# path = os.path.abspath(sys.argv[1])
# with open(path, "rb") as f:
#     data = tomlmod.load(f)

# deps = data.get("project", {}).get("dependencies", [])
# for d in deps or []:
#     print(d)
# PY

#   if [[ ! -s "$tmp_req" ]]; then
#     echo "Warning: No [project].dependencies found in $toml_path"
#   fi
#   echo "==> Installing dependencies from [project].dependencies in $toml_path"
#   uv pip install -p "$PYTHON" -r "$tmp_req"
#   rm -f "$tmp_req"
# }

case "$DEPS_FILE" in
  *.txt)
    echo "==> Installing dependencies from requirements file: $DEPS_FILE"
    uv pip install -p "$PYTHON" -r "$DEPS_FILE"
    ;;
  *pyproject.toml)
    install_from_pyproject "$DEPS_FILE"
    ;;
  *)
    echo "Error: Unsupported dependencies file type: $DEPS_FILE"
    echo "Provide either a requirements.txt or a pyproject.toml."
    exit 1
    ;;
esac

echo ""
echo "=================================================="
echo "✅ Environment setup complete"
echo "Task:     $TASK_NAME"
echo "Venv:     $VENV_DIR"
echo "Python:   $PYTHON"
echo "=================================================="
echo ""
echo "Use this in your config or runner (requires LocalJobConfig.python_path support):"
echo ""
echo "from shinka.launch import LocalJobConfig"
echo "job_config = LocalJobConfig("
echo "    eval_program_path=\"evaluate.py\","
echo "    python_path=\"$PYTHON\""
echo ")"
echo ""