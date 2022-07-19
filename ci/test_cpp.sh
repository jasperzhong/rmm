#!/bin/bash

set -euo pipefail

# Check environment
source ci/check_env.sh

# GPU Test Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

gpuci_mamba_retry install \
  -c "${CPP_CHANNEL}" \
  rmm librmm librmm-tests

TESTRESULTS_DIR=test-results
mkdir -p ${TESTRESULTS_DIR}
SUITEERROR=0

gpuci_logger "Check GPU usage"
nvidia-smi

set +e

gpuci_logger "Running googletests"
# run gtests from librmm-tests package
for gt in "$CONDA_PREFIX/bin/gtests/librmm/"* ; do
    ${gt} --gtest_output=xml:${TESTRESULTS_DIR}/
    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: GTest ${gt}"
    fi
done

exit ${SUITEERROR}
