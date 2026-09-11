#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel)"
pr_head_sha="$(git -C "${repo_root}" rev-parse HEAD)"
image="${IMAGE:-rosidl-buffer-backends:pr-11-resolute}"

docker build \
  --pull \
  --file "${script_dir}/Dockerfile" \
  --build-arg "PR_HEAD_SHA=${pr_head_sha}" \
  --tag "${image}" \
  "${repo_root}"
