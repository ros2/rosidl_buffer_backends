#!/usr/bin/env bash
set -eo pipefail

source_root="${1:?source root is required}"
artifact_dir="${2:?artifact directory is required}"
ros_distro="${ROS_DISTRO:-rolling}"
os_codename="${OS_CODENAME:-resolute}"

source "/opt/ros/${ros_distro}/setup.bash"

# Bloom resolves dependencies through rosdep/rosdistro metadata, including for
# packages already installed in the build root. These packages are not released
# yet, so provide the mapping that their rosdistro release entries will supply.
local_rosdep_source="${source_root}/docker/resolute/debian/local-packages.yaml"
printf 'yaml file://%s\n' "${local_rosdep_source}" \
  > /etc/ros/rosdep/sources.list.d/00-rosidl-buffer-backends-local.list
rosdep update

standalone_package_paths=(
  "onnxruntime_vendor/onnxruntime_core_vendor"
  "onnxruntime_vendor/python_onnxruntime_vendor"
)

package_paths=(
  "tensor_msgs"
  "cuda_buffer_backend/cuda_buffer_backend_msgs"
  "cuda_buffer_backend/cuda_buffer"
  "cuda_buffer_backend/cuda_buffer_backend"
  "cuda_buffer_backend/cuda_buffer_py"
  "onnxruntime_vendor/onnxruntime_cuda_vendor"
  "onnxruntime_vendor/python_onnxruntime_cuda_vendor"
  "onnxruntime_conversions/onnxruntime_conversions"
  "onnxruntime_conversions/onnxruntime_conversions_cpu"
  "onnxruntime_conversions/onnxruntime_conversions_cuda"
  "onnxruntime_conversions/onnxruntime_conversions_py"
  "onnxruntime_conversions/onnxruntime_conversions_py_cpu"
  "onnxruntime_conversions/onnxruntime_conversions_py_cuda"
)

rosdep_paths=()
for package_path in "${standalone_package_paths[@]}" "${package_paths[@]}"; do
  rosdep_paths+=("${source_root}/${package_path}")
done
rosdep_paths+=(
  "${source_root}/docker/resolute/debian/installed_tests/onnxruntime_conversions_installed_tests"
)

apt-get update
rosdep install \
  --from-paths "${rosdep_paths[@]}" \
  --ignore-src \
  --rosdistro "${ros_distro}" \
  --as-root apt:true \
  -y

command -v nvcc
if python3 -c 'import onnxruntime' 2>/dev/null; then
  echo "Python ONNX Runtime was installed unexpectedly" >&2
  exit 1
fi

export CUDACXX
CUDACXX="$(command -v nvcc)"

mkdir -p "${artifact_dir}" /tmp/ros-debian-builds

build_and_install_debian() {
  local package_path="$1"
  local install_package="${2:-true}"
  local package_name
  package_name="$(basename "${package_path}")"
  local build_parent="/tmp/ros-debian-builds/${package_name}"
  local build_source="${build_parent}/source"

  mkdir -p "${build_source}"
  cp -a "${source_root}/${package_path}/." "${build_source}/"

  pushd "${build_source}"
  bloom-generate rosdebian \
    --ros-distro "${ros_distro}" \
    --os-name ubuntu \
    --os-version "${os_codename}"
  DEB_BUILD_OPTIONS=nocheck dpkg-buildpackage -us -uc -b
  popd

  local main_debian=""
  while IFS= read -r debian; do
    cp -a "${debian}" "${artifact_dir}/"
    if [[ "${debian}" != *-dbgsym_*.deb ]]; then
      main_debian="${debian}"
    fi
  done < <(find "${build_parent}" -maxdepth 1 -type f -name '*.deb' | sort)

  if [[ -z "${main_debian}" ]]; then
    echo "No binary Debian was generated for ${package_name}" >&2
    exit 1
  fi

  if [[ "${install_package}" == true ]]; then
    apt-get install -y "${main_debian}"
  fi
}

for package_path in "${standalone_package_paths[@]}"; do
  build_and_install_debian "${package_path}" false
done
for package_path in "${package_paths[@]}"; do
  build_and_install_debian "${package_path}"
done

cached_debians=(/var/cache/apt/archives/*.deb)
if [[ -e "${cached_debians[0]}" ]]; then
  cp -a "${cached_debians[@]}" "${artifact_dir}/"
fi

cp -a /root/.ros/rosdep/sources.cache "${artifact_dir}/rosdep-sources-cache"

pushd "${artifact_dir}"
dpkg-scanpackages . /dev/null > Packages
gzip -9cn Packages > Packages.gz
sha256sum ./*.deb > SHA256SUMS
: > MANIFEST.tsv
for debian in ./*.deb; do
  dpkg-deb --showformat='${Package}\t${Version}\t${Architecture}\n' \
    --show "${debian}" >> MANIFEST.tsv
done
popd

echo "Generated Debian packages:"
cat "${artifact_dir}/MANIFEST.tsv"
