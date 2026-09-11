#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/../../.." && pwd)"
artifact_dir="${script_dir}/artifacts"
report_dir="${script_dir}/reports"
buildbase_tag="rosidl-buffer-backends:resolute-debian-buildbase"
test_image="ros:rolling-ros-base-resolute"

mkdir -p "${artifact_dir}" "${report_dir}"
docker run --rm \
  --mount "type=bind,src=${artifact_dir},dst=/artifacts" \
  --entrypoint find \
  "${test_image}" \
  /artifacts -mindepth 1 -delete

docker build \
  --file "${script_dir}/Dockerfile.buildfarm" \
  --target buildbase \
  --tag "${buildbase_tag}" \
  "${repository_root}"

docker run --rm \
  --mount "type=bind,src=${repository_root},dst=/source/rosidl_buffer_backends,readonly" \
  --mount "type=bind,src=${artifact_dir},dst=/artifacts" \
  --entrypoint bash \
  "${buildbase_tag}" \
  /source/rosidl_buffer_backends/docker/resolute/debian/build-debians.sh \
  /source/rosidl_buffer_backends /artifacts

test -s "${artifact_dir}/Packages"
test -s "${artifact_dir}/MANIFEST.tsv"
cat "${artifact_dir}/MANIFEST.tsv"

production_packages=(
  onnxruntime-conversions
  onnxruntime-conversions-cpu
  onnxruntime-conversions-cuda
  onnxruntime-conversions-py
  onnxruntime-conversions-py-cpu
  onnxruntime-conversions-py-cuda
)
for package in "${production_packages[@]}"; do
  debian="$(find "${artifact_dir}" -maxdepth 1 -type f \
    -name "ros-rolling-${package}_*.deb" ! -name '*-dbgsym_*' -print -quit)"
  test -n "${debian}"
  if dpkg-deb --contents "${debian}" |
    grep -E '/(test|tests)/|/test_[^/]*$' >/dev/null
  then
    echo "Production Debian unexpectedly ships tests: ${debian}" >&2
    exit 1
  fi
done
echo 'production-debians-test-free=pass'

docker run --rm --gpus all \
  --mount "type=bind,src=${artifact_dir},dst=/var/local/ros-debs,readonly" \
  --mount "type=bind,src=${repository_root},dst=/source/rosidl_buffer_backends,readonly" \
  --mount "type=bind,src=${report_dir},dst=/reports" \
  --mount "type=bind,src=${script_dir}/run-consumer-test.sh,dst=/usr/local/bin/run-consumer-test.sh,readonly" \
  --mount "type=bind,src=${script_dir}/test-installed-debians.sh,dst=/usr/local/bin/test-installed-debians.sh,readonly" \
  --entrypoint bash \
  "${test_image}" \
  /usr/local/bin/run-consumer-test.sh
