#!/bin/bash

set -euo pipefail

conda config --set default_threads "${PARALLEL_LEVEL}"

# Update env vars
source rapids-env-update

# TODO: Move in recipe build?
export CMAKE_GENERATOR=Ninja

# TODO: Move to job config
export CUDA=11.5

# Check env
source ci/check_env.sh

################################################################################
# BUILD - Conda package builds (LIBRMM)
################################################################################

gpuci_logger "Begin cpp build"

gpuci_mamba_retry mambabuild conda/recipes/librmm

rapids-upload-conda-to-s3 cpp
