#!/bin/bash

# Thomas Lake
# Bash script to check if all files in a directory are of a specific type

# Base directory containing your training folders
base_dir="/my_image_directory/"

# Output file to store the list of non-jpg/jpeg files
output_file="non_jpg_files.txt"

# Create or clear the output file
> "$output_file"

# Loop through all directories in the base directory
for dir in "$base_dir"/*; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        # Find files not ending in .jpg or .jpeg and append to output file
        find "$dir" -type f ! \( -name '*.jpg' -o -name '*.jpeg' \) >> "$output_file"
    fi
done

# Print a message with the number of non-jpg/jpeg files found
echo "Non-jpg/jpeg files have been listed in $output_file"
