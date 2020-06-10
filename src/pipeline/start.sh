#!/bin/bash
export MYDIR="$(dirname "$(realpath "$0")")"

python ${MYDIR}/pipeline.py \
    ${MYDIR}/volumes/in_data/records.json \
    ${MYDIR}/volumes/in_data/config.yml \
    ${MYDIR}/volumes/out_data/processed_data \
    ${MYDIR}/volumes/out_data/topic_models \
    ${MYDIR}/volumes/out_data/models \
    -loglevel ${LOGLEVEL}
