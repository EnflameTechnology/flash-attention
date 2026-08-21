#!/bin/bash
# Copyright 2024 Enflame. All Rights Reserved.
#
set -eu -o pipefail
SCRIPT_DIR=$(dirname $(realpath $0))
project_dir=${SCRIPT_DIR}/..
BUILD_ROOT_DIR=$(pwd)
set -x
ARCH=$(uname -m)

# run in docker container
# registry-egc.enflame-tech.com/artifacts/public_pytorch:v2.11.0-TR3.8.106-ubuntu2204

function ci_build() {
  # cd ${project_name}
  cd ${project_dir}
  # sudo python3 -m pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com
  python3 setup.py bdist_wheel
}

function main() {
  $build_job_name
}

# export project_name=${project_name:-"attention"}

build_job_name=${1:-ci_build}

main "$@"
exit $?
         