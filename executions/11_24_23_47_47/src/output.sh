#!/bin/bash
BASEDIR=$PWD
set +x
. .colors.sh
set -e
if [ ! -f results/results.nc ]; then
    echo -e "$(c R)[error] The model has not generated the output results/results.nc"
    exit 1
else
    echo -e "$(c G )[success] The model has generated the output results/results.nc"
    mv results/results.nc ${OUTPUTS1}
fi
if [ ! -f results/results.mp4 ]; then
    echo -e "$(c R)[error] The model has not generated the output results/results.mp4"
    exit 1
else
    echo -e "$(c G )[success] The model has generated the output results/results.mp4"
    mv results/results.mp4 ${OUTPUTS2}
fi
