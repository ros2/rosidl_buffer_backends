#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

! command -v nvcc
! python3 -c 'import torch' 2>/dev/null
! ldconfig -p | grep -E 'libtorch|libcudart'

rm -f /etc/apt/sources.list.d/ubuntu.sources \
  /etc/apt/sources.list.d/ros2.sources
echo 'deb [trusted=yes] file:/var/local/ros-debs ./' \
  > /etc/apt/sources.list.d/rosidl-buffer-backends-local.list
cat > /etc/apt/preferences.d/rosidl-buffer-backends-local <<'EOF'
Package: ros-rolling-*
Pin: version 0.1.2-0resolute
Pin-Priority: 1001
EOF
apt-get update
apt-get install --no-install-recommends -y \
  ros-rolling-torch-conversions \
  ros-rolling-torch-conversions-cpu \
  ros-rolling-torch-conversions-py \
  ros-rolling-torch-conversions-py-cpu
! command -v nvcc
! dpkg-query -W 'nvidia-cuda*' >/dev/null 2>&1
! dpkg-query -W ros-rolling-torch-conversions-cuda
! dpkg-query -W ros-rolling-torch-conversions-py-cuda

exec bash /usr/local/bin/test-installed-debians.sh \
  /source/rosidl_buffer_backends \
  /reports/torch-debian-installed-tests.log
