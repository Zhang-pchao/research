#!/bin/bash

# Define file paths
input_file="../COLVAR_tmp"
output_file="./COLVAR_tmp"

# Remove the output file if it already exists
if [ -f "$output_file" ]; then
    rm "$output_file"
fi

# Add the header to the output file
echo "#! FIELDS time s05 d05 adz opes.bias" > "$output_file"

# Filter lines and append to the output file
awk '{
    # Skip lines starting with #
    if (/^#/) next;

    # Extract $4 as numerical values
    col4 = $4 + 0;

    # Apply conditions to skip specific lines
    if ((col4 < 5) ||
	    (col4 > 12.15))
        next;

    # Print lines that do not meet the skip conditions
    print;
}' "$input_file" >> "$output_file"

