#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# add lucifer to PYTHONPATH, also add cruise to PYTHONPATH if using scm
export PYTHONPATH=$SCRIPT_DIR:/opt/tiger/cruise:$PYTHONPATH
bash setup_cruise.sh

# show all configs with defaults and user override first
python3 $@ --print_config

# run it
if ! command -v TORCHRUN &> /dev/null
then
    echo "==<TORCHRUN could not be found, use cruise included script>=="
    #--rdzv_endpoint=127.0.0.1:30000
    /opt/tiger/cruise/cruise/tools/TORCHRUN $@
    exit
else
    TORCHRUN $@
fi
