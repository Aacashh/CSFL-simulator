#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TARGET="${REPO_ROOT}/external/GenerativeFL"

mkdir -p "${REPO_ROOT}/external"
if [[ -d "${TARGET}/.git" ]]; then
  printf 'Official FedGCS repository is already present: %s\n' "${TARGET}"
else
  git clone --depth 1 https://github.com/zhiyuan-ning/GenerativeFL.git "${TARGET}"
fi

printf '\nPrepared the official FedGCS codebase.\n'
printf 'Read %s before running or importing a FedGCS result.\n' \
  "${SCRIPT_DIR}/official_fedgcs_protocol.md"

