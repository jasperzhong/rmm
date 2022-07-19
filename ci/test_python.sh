#!/bin/bash

set -euo pipefail

#TODO: Remove
. /opt/conda/etc/profile.d/conda.sh
conda activate base

# Check environment
source ci/check_env.sh

# GPU Test Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

gpuci_mamba_retry install \
  -c "${CPP_CHANNEL}" \
  -c "${PYTHON_CHANNEL}" \
  rmm librmm

gpuci_mamba_retry install pytest pytest-cov

TESTRESULTS_DIR=test-results
mkdir -p ${TESTRESULTS_DIR}
SUITEERROR=0

gpuci_logger "Check GPU usage"
nvidia-smi

cd python

set +e

gpuci_logger "pytest rmm"
py.test --cache-clear --junitxml=test-results/junit-rmm.xml -v --cov-config=.coveragerc --cov=rmm --cov-report=xml:python/rmm-coverage.xml --cov-report term
exitcode=$?
if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: 1 or more tests in /rmm/python"
fi

exit ${SUITEERROR}
