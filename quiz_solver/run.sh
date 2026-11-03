#!/usr/bin/env bash
# Convenience launcher: uses .env if present, otherwise asks for the phone IP.
set -e
cd "$(dirname "$0")"
if [ ! -f .env ]; then
    read -rp "Phone IP (e.g. 192.168.1.50): " PHONE_IP
    PROVIDER="${PROVIDER:-gemini}"
    STREAM="http://${PHONE_IP}:8080/video"
else
    PROVIDER="${PROVIDER:-gemini}"
    STREAM="${PHONE_STREAM_URL:-http://192.168.1.100:8080/video}"
fi
python3 main.py --stream-url "$STREAM" --provider "$PROVIDER" --auto "$@"
