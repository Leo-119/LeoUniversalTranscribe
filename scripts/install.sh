#!/bin/zsh
set -e

cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Error: macOS is required."
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Error: Apple Silicon is required."
    exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
    echo "Error: Homebrew is required."
    exit 1
fi

if ! command -v python3.13 >/dev/null 2>&1; then
    echo "Error: Python 3.13 is required."
    exit 1
fi

brew install ffmpeg yt-dlp

python3.13 -m venv .venv

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo ""
echo "Installation completed."
echo ""
echo "Run:"
echo ".venv/bin/python src/LeoUniversalTranscribe.py --help"
