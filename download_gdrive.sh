#!/usr/bin/env bash
#
# download_gdrive.sh — download a PUBLIC Google Drive folder into the current dir.
#
# Usage:
#   ./download_gdrive.sh                      # uses the default ECRot folder below
#   ./download_gdrive.sh <google_drive_url>   # any public folder or file URL
#
# The Drive folder/file must be shared as "Anyone with the link". gdown handles
# Drive's large-file virus-scan confirmation automatically.
#
set -euo pipefail

URL="${1:-https://drive.google.com/drive/folders/1w4AEgAAlrUZORWa6ajPDGoqkTv8TozNm}"

echo "Installing/updating gdown ..."
python3 -m pip install --user -q -U gdown

echo "Downloading into: $(pwd)"
python3 - "$URL" <<'PY'
import sys, gdown
url = sys.argv[1]
if "/folders/" in url:
    gdown.download_folder(url=url, output=".", quiet=False,
                          use_cookies=False, remaining_ok=True)
else:
    gdown.download(url=url, output=".", quiet=False, fuzzy=True, use_cookies=False)
print("\nDone -> files are in the current directory")
PY
