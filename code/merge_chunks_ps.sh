#!/bin/bash

# Usage: ./merge_chunks.sh /path/to/folder

if [ $# -ne 1 ]; then
    echo "Usage: $0 /path/to/folder"
    exit 1
fi

FOLDER="$1"

# Extract the folder name (e.g., "rosbag2_2026_03_30-11_46_39")
# The ${FOLDER%/} syntax removes a trailing slash if it exists
FOLDER_NAME=$(basename "${FOLDER%/}")
OUTPUT_NAME="${FOLDER_NAME}_merged.mcap"

cd "$FOLDER" || { echo "Cannot enter folder: $FOLDER"; exit 1; }

# Find all files matching rosbag2_*
FILES=(rosbag2_*)
if [ ${#FILES[@]} -eq 0 ]; then
    echo "No rosbag2_* files found in $FOLDER"
    exit 1
fi

# Merge the files
mcap merge --allow-duplicate-metadata "${FILES[@]}" -o "$OUTPUT_NAME"

echo "Merged output saved as $FOLDER/$OUTPUT_NAME"
