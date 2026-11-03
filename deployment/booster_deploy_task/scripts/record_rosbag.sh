#!/usr/bin/env bash
# ROS 2 Humble recording only. Does not launch the policy or send motor commands.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
qos_file="${script_dir}/../config/rosbag_qos.yaml"
if [[ $# -gt 1 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: bash %s [NEW_BAG_DIRECTORY]\nStop recording with Ctrl+C; wait for the recorder to finish.\n' "$0"
    exit 0
fi
if ! command -v ros2 >/dev/null 2>&1; then
    printf 'ros2 not found. Source ROS Humble and the Booster message workspace first.\n' >&2
    exit 1
fi
if [[ "${ROS_DISTRO:-}" != "humble" ]]; then
    printf 'Expected ROS_DISTRO=humble; source the robot ROS environment first.\n' >&2
    exit 1
fi
output="${1:-recordings/team15_$(date -u +%Y%m%dT%H%M%S)_$$_${RANDOM}}"
if [[ -z "$output" || -e "$output" || -L "$output" ]]; then
    printf 'Refusing existing/empty output path: %s\n' "$output" >&2
    exit 1
fi
[[ -f "$qos_file" ]] || { printf 'Missing QoS file: %s\n' "$qos_file" >&2; exit 1; }
mkdir -p -- "$(dirname -- "$output")"
printf 'Recording to: %s\nStop with Ctrl+C. This process does not start robot control.\n' "$output"
exec ros2 bag record --storage sqlite3 --output "$output" \
    --qos-profile-overrides-path "$qos_file" \
    /low_state /joint_ctrl /remote_controller_state /rosout
