#!/bin/bash
set -e
SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
CAM=/dev/video0
OPENCLAW_DIR=${HOME}/.openclaw
OUTDIR=${OPENCLAW_DIR}/workspace/camera_presence
# neutral
v4l2-ctl -d ${CAM} --set-ctrl brightness=128 || true
cd ${SCRIPT_DIR} && uv run python presence_observer.py --camera-device ${CAM} --output-dir ${OUTDIR}
# read result
JSON=${OUTDIR}/latest_report.json
if [ -f "${JSON}" ]; then
  PRESENCE=$(jq -r '.summary.presence' "${JSON}" 2>/dev/null || echo "unknown")
  if [ "$PRESENCE" = "present" ] || [ "$PRESENCE" = "possible_present" ]; then
    # smile for both confirmed and possible presence
    v4l2-ctl -d ${CAM} --set-ctrl brightness=255 || true
  else
    # sad briefly then neutral
    v4l2-ctl -d ${CAM} --set-ctrl brightness=0 || true
    sleep 3
    v4l2-ctl -d ${CAM} --set-ctrl brightness=128 || true
  fi
fi
