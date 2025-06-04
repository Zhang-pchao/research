import os

def find_xyz_files(base_directory, endstr='_tmp'):
    """Find all files with the specified ending in the base directory and its subdirectories."""
    xyz_files = []
    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith(endstr):
                xyz_files.append(os.path.join(root, file))
    return xyz_files

def concatenate_xyz_files(xyz_files, output_file):
    """Concatenate all files into one output file."""
    with open(output_file, 'w') as outfile:
        # Insert the specified line at the beginning of the output file
        outfile.write('#! FIELDS time s05 d05 opes.bias\n')
        
        for xyz_file in xyz_files:
            with open(xyz_file, 'r') as infile:
                outfile.write(infile.read())
                #outfile.write('\n')  # Add a newline to separate files

def main():
    # Automatically find all directories in the current working directory
    base_directory = os.getcwd()

    # Find all files with the specified ending
    xyz_files = find_xyz_files(base_directory)

    # Print the names of the files found
    print("Found files:")
    for xyz_file in xyz_files:
        print(xyz_file)

    # Concatenate all files into one output file
    output_file = 'COLVAR_tmp'
    concatenate_xyz_files(xyz_files, output_file)
    print(f"All files have been concatenated into {output_file}")

if __name__ == '__main__':
    main()