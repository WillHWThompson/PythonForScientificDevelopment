#!/bin/bash
# tools/sync_vacc.sh
REMOTE_PATH="vacc:/users/w/t/wthomps3/CSDS/side_projects/PythonForScientificDevelopment/results/"
LOCAL_PATH="./results/"

echo "--- Syncing Research Data from VACC (Excluding Weights) ---"
rsync -avP \
    --include="*/" \
    --include="*.parquet" \
    --include="*.json" \
    --include="*.log" \
    --exclude="*.pt" \
    --exclude="*" \
    "$REMOTE_PATH" "$LOCAL_PATH"
echo "--- Sync Complete ---"
