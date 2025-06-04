#!/bin/bash

# Store the current directory
current_dir=$(pwd)

# Find all subdirectories containing the file 'COLVAR_tmp'
find . -type f -name 'COLVAR_tmp' | while read -r file; do
    # Get the directory containing the file
    dir=$(dirname "$file")

    # Insert the specified line into the first line of 'COLVAR_tmp'
    echo '#! FIELDS time s05 d05 opes.bias' | cat - "$file" > temp && mv temp "$file"

    # Check if 'cvfig' already exists in the target directory
    if [ ! -d "$dir/cvfig" ]; then
        # Copy the 'cvfig' directory to the current subdirectory
        cp -r "$current_dir/cvfig" "$dir"
		# Change to the directory where 'cvfig' is or was copied
		cd "$dir/cvfig" || exit
		# Execute the command
		sh q.sh				
    else
        echo "'cvfig' already exists in $dir, skipping copy."
    fi
    
    # Change back to the original directory
    cd "$current_dir" || exit
done
