#!/bin/bash

BASE="datasets/outputs/outputs_4"
GIVEN_FOLDER="$1"

TARGET_DIR="$BASE/$GIVEN_FOLDER"

if [ ! -d "$TARGET_DIR" ]; then
  echo "Folder not found: $TARGET_DIR"
  exit 1
fi

# IMPORTANT: no commas in bash arrays
#SKIP_LIST=("3" "13" "23" "33" "37" "55" "56" "adult" "compas" "credit" "german" "law" "oulad" "student")

echo "Processing: $TARGET_DIR"

for dataset_dir in "$TARGET_DIR"/*; do
  if [ -d "$dataset_dir" ]; then

    dataset_name=$(basename "$dataset_dir")

    skip=false
    for skip_name in "${SKIP_LIST[@]}"; do
      if [ "$dataset_name" == "$skip_name" ]; then
        skip=true
        break
      fi
    done

    if [ "$skip" == true ]; then
      echo "⏭️ Skipping dataset '${dataset_name}' (on skip list)"
      continue
    fi

    output_file="$TARGET_DIR/${dataset_name}.tar.gz"

    tar -czf "$output_file" -C "$TARGET_DIR" "$dataset_name"

    echo "✅ ${dataset_name} done"

  fi
done

echo "🎉 All datasets done"