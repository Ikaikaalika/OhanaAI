#!/bin/bash
# Download WikiTree GEDCOM dump to an external drive.
# Usage: ./scripts/download_wikitree.sh /Volumes/OhanaData/wikitree
set -euo pipefail
TARGET_DIR="${1:-/Volumes/OhanaData/wikitree}"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"
ARCHIVE_URL="https://wiki.wikitree.com/treeExport/test/wikitree_dump.7z"
ARCHIVE_NAME="wikitree_dump.7z"
if [ ! -f "$ARCHIVE_NAME" ]; then
  curl -O "$ARCHIVE_URL"
fi
if ! command -v 7z >/dev/null 2>&1; then
  echo "7z is required. Install via 'brew install p7zip' (macOS) or your package manager." >&2
  exit 1
fi
7z x -aoa "$ARCHIVE_NAME"
echo "Download and extraction complete in $TARGET_DIR"
