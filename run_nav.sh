#!/bin/bash
mkdir -p output_dirs
export PYTHONPATH=./:./hm3d-online:./hm3d-online/FastSAM
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
sleep 1
# Run the Python script
while true; do
    python3 hm3d-online/ovon-nav.py
    if [ $? -eq 0 ]; then
        break
    fi
done        