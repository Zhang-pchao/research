import os
import re
import numpy as np
import sys

def process_single_xyz(input_path, output_dir):
    """Process a single XYZ file and generate a DPData format directory."""
    # Create the output directory structure
    os.makedirs(output_dir, exist_ok=True)
    set_dir = os.path.join(output_dir, 'set.000')
    os.makedirs(set_dir, exist_ok=True)

    # Define the fixed element mapping
    type_map = ['H', 'O']
    with open(os.path.join(output_dir, 'type_map.raw'), 'w') as f:
        for elem in type_map:
            f.write(f"{elem}\n")

    # Read and parse the XYZ file
    frames = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        current_line = 0
        total_lines = len(lines)
        
        while current_line < total_lines:
            # Read the number of atoms
            try:
                n_atoms = int(lines[current_line].strip())
            except:
                raise ValueError(f"Invalid number of atoms @ line {current_line+1}")
            current_line += 1
            
            # Parse the info line
            info_line = lines[current_line].strip()
            current_line += 1
            info = {}
            pattern = re.compile(r'(\w+)=("[^"]*"|\S+)')
            for key, value in re.findall(pattern, info_line):
                if value.startswith('"'):
                    value = value[1:-1]
                info[key] = value
            
            # Extract key information
            try:
                energy = float(info['energy'])
                virial = np.array(list(map(float, info['virial'].split()))).reshape(3, 3)
                lattice = np.array(list(map(float, info['Lattice'].split()))).reshape(3, 3)
            except KeyError as e:
                raise ValueError(f"Missing required field {e} @ line {current_line}")
            
            # Read atom data
            atoms = []
            for _ in range(n_atoms):
                if current_line >= total_lines:
                    raise ValueError("Unexpected end of file")
                parts = lines[current_line].strip().split()
                current_line += 1
                if len(parts) < 7:
                    raise ValueError(f"Incomplete atom data @ line {current_line}")
                species = parts[0]
                pos = list(map(float, parts[1:4]))
                force = list(map(float, parts[4:7]))
                atoms.append({'species': species, 'pos': pos, 'force': force})
            
            frames.append({
                'n_atoms': n_atoms,
                'energy': energy,
                'virial': virial,
                'lattice': lattice,
                'atoms': atoms
            })

    # Generate type.raw
    first_atoms = frames[0]['atoms']
    type_list = []
    for atom in first_atoms:
        try:
            type_idx = type_map.index(atom['species'])
        except ValueError:
            raise ValueError(f"Element {atom['species']} is not in the predefined type_map")
        type_list.append(type_idx)
    with open(os.path.join(output_dir, 'type.raw'), 'w') as f:
        for t in type_list:
            f.write(f"{t}\n")

    # Validate consistency of atom types in subsequent frames
    for i, frame in enumerate(frames[1:]):
        frame_species = [a['species'] for a in frame['atoms']]
        if len(frame_species) != len([a['species'] for a in first_atoms]):
            raise ValueError(f"Inconsistent number of atoms at frame {i+2}")
        if frame_species != [a['species'] for a in first_atoms]:
            raise ValueError(f"Inconsistent atom type/order at frame {i+2}")

    # Collect all data
    energies = []
    virials = []
    boxes = []
    coords = []
    forces = []
    
    for frame in frames:
        energies.append(frame['energy'])
        virials.append(frame['virial'].flatten())  # Flatten to a 9-element vector
        boxes.append(frame['lattice'].flatten())   # Flatten to a 9-element vector
        # Collect and flatten coordinates and forces
        frame_coords = []
        frame_forces = []
        for atom in frame['atoms']:
            frame_coords.extend(atom['pos'])  # Append coordinates of each atom into a single vector
            frame_forces.extend(atom['force']) # Append forces of each atom into a single vector
        coords.append(frame_coords)
        forces.append(frame_forces)
    
    # Convert to numpy arrays
    energies_np = np.array(energies)
    virials_np = np.array(virials)      # Correct shape: (n_frames, 9)
    boxes_np = np.array(boxes)          # Correct shape: (n_frames, 9)
    coords_np = np.array(coords)        # Correct shape: (n_frames, n_atoms*3)
    forces_np = np.array(forces)        # Correct shape: (n_frames, n_atoms*3)

    # Write .raw files
    def write_raw(data, filename, fmt):
        with open(os.path.join(output_dir, filename), 'w') as f:
            for item in data:
                if isinstance(item, (np.ndarray, list)):
                    line = ' '.join([fmt % x for x in item])
                else:
                    line = fmt % item
                f.write(line + '\n')

    write_raw(energies, 'energy.raw', '%.15e')
    write_raw(virials_np, 'virial.raw', '%.15e')
    write_raw(boxes_np, 'box.raw', '%.15e')
    write_raw(coords_np, 'coord.raw', '%.15e')
    write_raw(forces_np, 'force.raw', '%.15e')

    # Save .npy files
    np.save(os.path.join(set_dir, 'energy.npy'), energies_np)
    np.save(os.path.join(set_dir, 'virial.npy'), virials_np)
    np.save(os.path.join(set_dir, 'box.npy'), boxes_np)
    np.save(os.path.join(set_dir, 'coord.npy'), coords_np)
    np.save(os.path.join(set_dir, 'force.npy'), forces_np)

    # Print the shapes of the saved data
    print(f"  Saved file shape info:")
    print(f"  energy.npy: {energies_np.shape}")
    print(f"  virial.npy: {virials_np.shape}")
    print(f"  box.npy: {boxes_np.shape}")
    print(f"  coord.npy: {coords_np.shape}")
    print(f"  force.npy: {forces_np.shape}")

def batch_process_files(input_files, output_folder='/home/dpdata_output'):
    """Process a specified list of XYZ files."""
    os.makedirs(output_folder, exist_ok=True)
    
    processed = 0
    for xyz_file in input_files:
        try:
            base_name = os.path.splitext(os.path.basename(xyz_file))[0]
            output_dir = os.path.join(output_folder, f"{base_name}_dpdata")
            
            print(f"Processing file: {xyz_file}")
            process_single_xyz(xyz_file, output_dir)
            processed += 1
            print(f"Successfully converted: {xyz_file} -> {output_dir}")
        except Exception as e:
            print(f"Error processing {xyz_file}: {str(e)}")
    
    print(f"Processing complete! Successfully converted {processed} files.")

def batch_convert(input_folder, output_folder='/home/dpdata_output'):
    """Batch process all XYZ files in a folder."""
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    xyz_files = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.xyz'):
            xyz_files.append(os.path.join(input_folder, filename))
    
    if not xyz_files:
        print(f"No XYZ files found in {input_folder}")
        return
    
    batch_process_files(xyz_files, output_folder)

if __name__ == '__main__':
    # Predefined file list
    predefined_files = [
        "63H2O.xyz",
        "64H2O.xyz",
    ]
    
    if len(sys.argv) < 2:
        print("Error: Input directory must be provided.")
        print("Usage: python xyz2dpdata.py <input_directory> [output_directory]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = '/home/dpdata_output'
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Construct full file paths
    full_path_files = []
    for filename in predefined_files:
        file_path = os.path.join(input_dir, filename)
        if os.path.exists(file_path):
            full_path_files.append(file_path)
        else:
            print(f"Warning: File not found {file_path}")
    
    if full_path_files:
        batch_process_files(full_path_files, output_dir)
    else:
        print(f"Specified files not found in {input_dir}, will process all xyz files in the directory.")
        batch_convert(input_dir, output_dir)