#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

! command -v nvcc
! python3 -c 'import onnxruntime' 2>/dev/null
! ldconfig -p | grep -E 'libonnxruntime|libcudart|libcudnn'

echo 'deb [trusted=yes] file:/var/local/ros-debs ./' \
  > /etc/apt/sources.list.d/rosidl-buffer-backends-local.list
cat > /etc/apt/preferences.d/rosidl-buffer-backends-local <<'EOF'
Package: ros-rolling-*
Pin: version 0.1.2-0resolute
Pin-Priority: 1001
EOF
apt-get update
apt-get install --no-install-recommends -y \
  ros-rolling-onnxruntime-conversions \
  ros-rolling-onnxruntime-conversions-cpu \
  ros-rolling-onnxruntime-conversions-py \
  ros-rolling-onnxruntime-conversions-py-cpu
! dpkg-query -W ros-rolling-onnxruntime-conversions-cuda
! dpkg-query -W ros-rolling-onnxruntime-conversions-py-cuda

exec bash /usr/local/bin/test-installed-debians.sh \
  /source/rosidl_buffer_backends \
  /reports/onnxruntime-debian-installed-tests.log
