#!/bin/bash
set -eu
mkdir -p /logs/verifier
if [ "$(cat /tmp/autoeval-proof.txt 2>/dev/null || true)" = AUTOEVAL_PIPELINE_OK ]; then
    printf '1\n' > /logs/verifier/reward.txt
else
    printf '0\n' > /logs/verifier/reward.txt
fi
