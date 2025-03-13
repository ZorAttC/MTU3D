#!/bin/bash
sleep 1800
# Run the Python script
while true; do
    python3 hm3d-online/eqa-traj.py
    if [ $? -eq 0 ]; then
        break
    fi
done        