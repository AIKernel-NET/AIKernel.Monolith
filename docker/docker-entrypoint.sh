#!/usr/bin/env sh
set -eu

if [ "${1:-}" = "--print-sample-config" ]; then
  exec /aikernel/AIKernel.Monolith --print-sample-config
fi

missing=""
for path in "${AIKERNEL_CONFIG_ROOT:-}" "${AIKERNEL_MODEL_ROOT:-}" "${AIKERNEL_VFS_ROOT:-}" "${AIKERNEL_CAPABILITY_ROOT:-}"; do
  if [ -z "$path" ] || [ ! -d "$path" ]; then
    missing="true"
  fi
done

if [ "$missing" = "true" ]; then
  echo "AIKernel.Monolith requires external mounts for config, models, VFS, and capabilities." >&2
  echo "Set AIKERNEL_CONFIG_ROOT, AIKERNEL_MODEL_ROOT, AIKERNEL_VFS_ROOT, and AIKERNEL_CAPABILITY_ROOT." >&2
  exit 2
fi

exec /aikernel/AIKernel.Monolith \
  --config-root="${AIKERNEL_CONFIG_ROOT}" \
  --model-root="${AIKERNEL_MODEL_ROOT}" \
  --vfs-root="${AIKERNEL_VFS_ROOT}" \
  --capability-root="${AIKERNEL_CAPABILITY_ROOT}" \
  "$@"
