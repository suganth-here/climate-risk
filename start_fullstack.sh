#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
INSIDE_NIX_SHELL="${1:-}"

append_lib_dir() {
  local dir="$1"
  if [[ -n "$dir" && -d "$dir" ]]; then
    LIB_DIRS+=("$dir")
  fi
}

prepare_runtime_libs() {
  LIB_DIRS=()
  append_lib_dir "/run/current-system/sw/lib"
  append_lib_dir "$HOME/.nix-profile/lib"
  append_lib_dir "/etc/profiles/per-user/${USER:-}/lib"

  if command -v gcc >/dev/null 2>&1; then
    local gcc_stdcpp
    gcc_stdcpp="$(gcc -print-file-name=libstdc++.so.6 2>/dev/null || true)"
    if [[ -n "$gcc_stdcpp" && "$gcc_stdcpp" != "libstdc++.so.6" ]]; then
      append_lib_dir "$(dirname "$gcc_stdcpp")"
    fi
  fi

  local soname libpath
  for soname in libstdc++.so.6 libz.so.1; do
    libpath="$(ls -1 /nix/store/*/lib/"$soname" 2>/dev/null | head -n 1 || true)"
    if [[ -n "$libpath" ]]; then
      append_lib_dir "$(dirname "$libpath")"
    fi
  done

  if [[ ${#LIB_DIRS[@]} -gt 0 ]]; then
    local joined=""
    local d
    for d in "${LIB_DIRS[@]}"; do
      if [[ ":${joined}:" != *":${d}:"* ]]; then
        joined="${joined:+${joined}:}${d}"
      fi
    done
    export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

prepare_runtime_libs

if [[ -x "$REPO_ROOT/env/bin/python" ]]; then
  PYTHON_CMD="$REPO_ROOT/env/bin/python"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_CMD="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_CMD="python3"
fi

if ! "$PYTHON_CMD" -c "import django, pandas, numpy" >/dev/null 2>&1; then
  if [[ "$INSIDE_NIX_SHELL" != "--inside-nix" ]] && command -v nix-shell >/dev/null 2>&1; then
    echo "Detected missing runtime libs; relaunching in nix-shell ..."
    exec nix-shell -p stdenv.cc.cc.lib --run "bash '$REPO_ROOT/start_fullstack.sh' --inside-nix"
  fi
  echo "Python dependencies are installed but not usable in this environment."
  echo "If you see 'libstdc++.so.6' errors, install your system C++ runtime and retry."
  exit 1
fi

echo "Starting Django backend on http://127.0.0.1:8000 ..."
"$PYTHON_CMD" backend/manage.py runserver 127.0.0.1:8000 &
BACKEND_PID=$!

cleanup() {
  if kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

echo "Starting React frontend on http://127.0.0.1:5173 ..."
cd "$REPO_ROOT/frontend"
if [[ -f "$REPO_ROOT/frontend/node_modules/vite/bin/vite.js" ]]; then
  node "$REPO_ROOT/frontend/node_modules/vite/bin/vite.js" --host 127.0.0.1 --port 5173
else
  npm run dev -- --host 127.0.0.1 --port 5173
fi
