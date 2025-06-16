#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# reset_state.sh – wipe indices & uploaded files for a fresh test run
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Paths to persistent artefacts
FAISS_INDEX="db_storage/faiss.index"
METADATA_DB="db_storage/metadata.db"
UPLOAD_DIR="opi_file_system"

echo "[reset_state] Removing FAISS index ($FAISS_INDEX) & metadata DB ($METADATA_DB) …"
rm -f "$FAISS_INDEX" "$METADATA_DB"

echo "[reset_state] Clearing uploaded files in $UPLOAD_DIR …"
if [ -d "$UPLOAD_DIR" ]; then
  find "$UPLOAD_DIR" -type f -not -name 'readme.md' -delete
fi

# ---------------------------------------------------------------------------
# Remove lingering System-V semaphores that Python/FAISS/Torch sometimes leak
# after segfaults. Only performed if the *ipcs* utility is available.
# ---------------------------------------------------------------------------
if command -v ipcs &>/dev/null && command -v ipcrm &>/dev/null; then
  echo "[reset_state] Cleaning stray semaphores owned by $(whoami) …"
  # Collect semaphore IDs owned by the current user (column 2 of ipcs -s)
  SEM_IDS=$(ipcs -s | awk -v user="$(whoami)" '$3==user {print $2}')
  if [ -n "$SEM_IDS" ]; then
    for sid in $SEM_IDS; do
      ipcrm -s "$sid" 2>/dev/null || true
    done
    echo "[reset_state] Removed $(echo "$SEM_IDS" | wc -w | tr -d ' ') semaphore(s)."
  else
    echo "[reset_state] No leaked semaphores detected."
  fi
else
  echo "[reset_state] ipcs/ipcrm not found – skipping semaphore cleanup."
fi

echo "[reset_state] Done. Fresh state ready." 