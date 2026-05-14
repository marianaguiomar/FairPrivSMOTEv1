#!/bin/bash

BASE="FairPrivSMOTEv1/datasets/outputs/outputs_4"GIVEN_FOLDER="$1"

TARGET_DIR="$BASE/$GIVEN_FOLDER"

if [ ! -d "$TARGET_DIR" ]; then
  echo "Folder not found: $TARGET_DIR"
  exit 1
fi

echo "Processing: $TARGET_DIR"

for dataset_dir in "$TARGET_DIR"/*; do
  if [ -d "$dataset_dir" ]; then

    dataset_name=$(basename "$dataset_dir")
    output_file="$TARGET_DIR/${dataset_name}.tar.gz"

    echo "Creating archive for $dataset_name -> $output_file"

    tar -czf "$output_file" -C "$TARGET_DIR" "$dataset_name"

  fi
done

echo "Done."