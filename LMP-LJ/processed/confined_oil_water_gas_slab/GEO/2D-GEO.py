#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Grid System Generator for LAMMPS Molecular Dynamics Simulations
===================================================================
This program builds the initial configuration of an oil-gas-water system.
It generates two types of output files:
1. LAMMPS data format (.dat)
2. XYZ format (.xyz)

Atom type definitions (six in total):
------------------------------------
Type 1 (sub):      Bottom fixed layer for lower boundary conditions
Type 2 (water):    Water phase for the bulk water atoms
Type 3 (gas):      Gas phase for the bubble region
Type 4 (oil-head): Oil head group, i.e., the hydrophilic end of oil molecules
Type 5 (oil-tail): Oil tail group, i.e., the hydrophobic end of oil molecules
Type 6 (top):      Top fixed layer for upper boundary conditions

Author: Ported and adapted from the C++ version
Date: 2025-11-04
"""

from typing import Tuple, List


class GridSystemGenerator:
    """3D grid system generator"""
    
    def __init__(self, x_size: int = 101, y_size: int = 5, z_size: int = 81, 
                 top_layer_offset: int = 10,
                 gas_x_start: int = 21, gas_x_width: int = 10,
                 oil_x_start: int = 61, oil_x_width: int = 20,
                 mixed_phase_z_start: int = 3, mixed_phase_z_thickness: int = 6):
        """
        Initialize the grid system.

        Parameters
        ----------
        x_size : int, default=101
            Number of grid points along X (recommended range: 10-200)
        y_size : int, default=5
            Number of grid points along Y (recommended range: 3-50)
        z_size : int, default=81
            Number of grid points along Z (recommended range: 10-200)
        top_layer_offset : int, default=10
            Downward offset from Z_SIZE for the top fixed layer
            Example: Z_SIZE=81, offset=10 -> top layer at z=69-70
                     Z_SIZE=81, offset=5  -> top layer at z=74-75
            Recommended range: 5-20
        gas_x_start : int, default=21
            Starting X position of the gas phase region
        gas_x_width : int, default=10
            Width of the gas phase region along X
        oil_x_start : int, default=61
            Starting X position of the oil phase region
        oil_x_width : int, default=20
            Width of the oil phase region along X
        mixed_phase_z_start : int, default=3
            Starting Z position of the mixed phase (gas+oil+water)
        mixed_phase_z_thickness : int, default=6
            Thickness of the mixed phase along Z
        """
        self.X = x_size
        self.Y = y_size
        self.Z = z_size
        self.top_offset = top_layer_offset
        
        # Gas and oil positional parameters
        self.gas_x_start = gas_x_start
        self.gas_x_width = gas_x_width
        self.oil_x_start = oil_x_start
        self.oil_x_width = oil_x_width
        self.mixed_z_start = mixed_phase_z_start
        self.mixed_z_thickness = mixed_phase_z_thickness
        
        # Initialize the 3D array that stores atom types (nested lists)
        self.ntype = [[[0 for _ in range(x_size)] for _ in range(y_size)] for _ in range(z_size)]
        
        # Initialize the 3D array that stores atom IDs (nested lists)
        self.id = [[[0 for _ in range(x_size)] for _ in range(y_size)] for _ in range(z_size)]
        
        # Store bond information [bond_id, [atom1_id, atom2_id]]
        self.bonds = []
        
        # Total number of atoms
        self.total_atoms = 0
        
    def define_regions(self, regions: List[dict] = None):
        """
        Assign atom types to different regions.

        Parameters
        ----------
        regions : List[dict], optional
            A list of region definitions. Each dictionary contains:
            - 'z_range': (z_min, z_max)
            - 'x_range': (x_min, x_max)
            - 'y_range': (y_min, y_max)
            - 'atom_type': int (atom type index)

            When omitted, the default configuration (matching the original C++
            program) is used.
        """
        if regions is None:
            # Use the default region configuration (parameters can be adjusted).
            # From bottom to top the system consists of: bottom fixed layer ->
            # mixed phase region -> bulk water phase -> top fixed layer.

            # Determine the Z range of the mixed phase region.
            mixed_z_end = self.mixed_z_start + self.mixed_z_thickness - 1

            # Determine the X ranges of the gas and oil regions.
            gas_x_end = self.gas_x_start + self.gas_x_width - 1
            oil_x_end = self.oil_x_start + self.oil_x_width - 1

            # Distance between gas and oil regions.
            gap_between_gas_oil = self.oil_x_start - gas_x_end - 1

            regions = [
                # Region 1: Bottom fixed layer (z=1-2) - type 1 (sub)
                {'z_range': (1, 2), 'x_range': (1, self.X-1), 'y_range': (1, self.Y-1), 'atom_type': 1},

                # Region 2: Mixed phase - left water section (before gas) - type 2 (water)
                {'z_range': (self.mixed_z_start, mixed_z_end),
                 'x_range': (1, self.gas_x_start - 1),
                 'y_range': (1, self.Y-1), 'atom_type': 2},

                # Region 3: Mixed phase - gas bubble (gas) - type 3 (gas)
                {'z_range': (self.mixed_z_start, mixed_z_end),
                 'x_range': (self.gas_x_start, gas_x_end),
                 'y_range': (1, self.Y-1), 'atom_type': 3},

                # Region 4: Mixed phase - central water (between gas and oil) - type 2 (water)
                {'z_range': (self.mixed_z_start, mixed_z_end),
                 'x_range': (gas_x_end + 1, self.oil_x_start - 1),
                 'y_range': (1, self.Y-1), 'atom_type': 2},

                # Region 5: Mixed phase - oil head layer (oil-head, y=1) - type 4 (oil-head)
                # Hydrophilic end of oil molecules, in contact with water.
                {'z_range': (self.mixed_z_start, mixed_z_end),
                 'x_range': (self.oil_x_start, oil_x_end),
                 'y_range': (1, 1), 'atom_type': 4},

                # Region 6: Mixed phase - oil tail layer (oil-tail, y=2-4) - type 5 (oil-tail)
                # Hydrophobic end of oil molecules forming the droplet interior.
                {'z_range': (self.mixed_z_start, mixed_z_end),
                 'x_range': (self.oil_x_start, oil_x_end),
                 'y_range': (2, self.Y-1), 'atom_type': 5},

                # Region 7: Mixed phase - right water section (after oil) - type 2 (water)
                {'z_range': (self.mixed_z_start, mixed_z_end),
                 'x_range': (oil_x_end + 1, self.X-1),
                 'y_range': (1, self.Y-1), 'atom_type': 2},

                # Region 8: Bulk water phase (z=9-50) - type 2 (water)
                {'z_range': (9, 50), 'x_range': (1, self.X-1), 'y_range': (1, self.Y-1), 'atom_type': 2},

                # Note: z=51 to the top offset region remains empty (no atoms).

                # Region 9: Top fixed layer - type 6 (top)
                # The position is controlled by the top_layer_offset parameter.
                {'z_range': (self.Z - self.top_offset - 2, self.Z - self.top_offset - 1),
                 'x_range': (1, self.X-1), 'y_range': (1, self.Y-1), 'atom_type': 6},
            ]

        # Iterate over the grid and assign atom types.
        for region in regions:
            z_min, z_max = region['z_range']
            x_min, x_max = region['x_range']
            y_min, y_max = region['y_range']
            atom_type = region['atom_type']

            for z in range(z_min, z_max + 1):
                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        if 0 <= z < self.Z and 0 <= y < self.Y and 0 <= x < self.X:
                            # Only count and assign IDs to previously empty sites.
                            if self.ntype[z][y][x] == 0:
                                self.total_atoms += 1
                                self.id[z][y][x] = self.total_atoms
                            # Update the atom type (allow overwriting).
                            self.ntype[z][y][x] = atom_type
    
    def create_bonds(self, bond_regions: List[dict] = None):
        """
        Create bonds between atoms.

        Parameters
        ----------
        bond_regions : List[dict], optional
            Bond definitions. Each dictionary contains:
            - 'z_range': (z_min, z_max)
            - 'x_range': (x_min, x_max)
            - 'y_range': (y_min, y_max)
            - 'direction': 'x', 'y', or 'z' (direction of the bond)

            When omitted, the default configuration from the C++ program is used.
        """
        if bond_regions is None:
            # Default configuration: create bonds along Y within the oil region.
            # Determine oil region dynamically using the configured parameters.
            mixed_z_end = self.mixed_z_start + self.mixed_z_thickness - 1
            oil_x_end = self.oil_x_start + self.oil_x_width - 1

            for z in range(self.mixed_z_start, mixed_z_end + 1):
                for x in range(self.oil_x_start, oil_x_end + 1):
                    for y in range(1, min(4, self.Y)):  # y=1 to 3 (or Y-1)
                        if y + 1 < self.Y:
                            atom1_id = self.id[z][y][x]
                            atom2_id = self.id[z][y+1][x]
                            if atom1_id > 0 and atom2_id > 0:
                                self.bonds.append([atom1_id, atom2_id])
        else:
            # Use custom bond configuration.
            for bond_region in bond_regions:
                z_min, z_max = bond_region['z_range']
                x_min, x_max = bond_region['x_range']
                y_min, y_max = bond_region['y_range']
                direction = bond_region.get('direction', 'y')
                
                for z in range(z_min, z_max + 1):
                    for x in range(x_min, x_max + 1):
                        for y in range(y_min, y_max + 1):
                            atom1_id = self.id[z][y][x]
                            if atom1_id > 0:
                                atom2_id = 0
                                if direction == 'x' and x + 1 < self.X:
                                    atom2_id = self.id[z][y][x+1]
                                elif direction == 'y' and y + 1 < self.Y:
                                    atom2_id = self.id[z][y+1][x]
                                elif direction == 'z' and z + 1 < self.Z:
                                    atom2_id = self.id[z+1][y][x]
                                
                                if atom2_id > 0:
                                    self.bonds.append([atom1_id, atom2_id])
    
    def write_lammps_data(self, filename: str = "oil_bubble_2.dat", num_atom_types: int = 7):
        """
        Write a LAMMPS data file.

        Parameters
        ----------
        filename : str, default="oil_bubble_2.dat"
            Output file name
        num_atom_types : int, default=7
            Total number of atom types
        """
        with open(filename, 'w') as f:
            # File header
            f.write("liquid data\n")
            f.write(f"\n{self.total_atoms} atoms\n")
            f.write(f"{len(self.bonds)} bonds\n")
            f.write(f"\n{num_atom_types} atom types\n")
            f.write("\n1 bond types\n")

            # Box dimensions
            f.write(f"0 {self.X - 1} xlo xhi\n")
            f.write(f"0 {self.Y - 1} ylo yhi\n")
            f.write(f"0 {self.Z - 1} zlo zhi\n")

            # Atom information
            f.write("\n\n\nAtoms\n")
            for z in range(1, self.Z):
                for y in range(1, self.Y):
                    for x in range(1, self.X):
                        if self.ntype[z][y][x] > 0 and self.id[z][y][x] >= 1:
                            atom_id = self.id[z][y][x]
                            atom_type = self.ntype[z][y][x]
                            # Coordinates (as floats)
                            coord_x = float(x)
                            coord_y = float(y)
                            coord_z = float(z)
                            # Format: atom_id  molecule_id  atom_type  x  y  z
                            f.write(f"\n{atom_id:4d}       {1:4d}     {atom_type:4d}     "
                                   f"{coord_x:8.5f}     {coord_y:8.5f}   {coord_z:8.5f}")

            # Bond information
            if len(self.bonds) > 0:
                f.write("\n\nBonds\n")
                for bond_id, (atom1, atom2) in enumerate(self.bonds, start=1):
                    # Format: bond_id  bond_type  atom1  atom2
                    f.write(f"\n{bond_id:4d}   {1:4d}       {atom1:4d}     {atom2:4d}")

        print(f"LAMMPS data file saved to: {filename}")
        print(f"  - Total atoms: {self.total_atoms}")
        print(f"  - Total bonds: {len(self.bonds)}")

    def write_xyz(self, filename: str = "oil_bubble_2.xyz"):
        """
        Write an XYZ format file for visualization.

        Parameters
        ----------
        filename : str, default="oil_bubble_2.xyz"
            Output file name
        """
        with open(filename, 'w') as f:
            # XYZ file header
            f.write(f"{self.total_atoms}           \n")
            f.write("Atoms.   Timestep:0")

            # Atomic coordinates
            for z in range(1, self.Z):
                for y in range(1, self.Y):
                    for x in range(1, self.X):
                        if self.ntype[z][y][x] > 0 and self.id[z][y][x] >= 1:
                            atom_type = self.ntype[z][y][x]
                            coord_x = float(x)
                            coord_y = float(y)
                            coord_z = float(z)
                            # Format: atom_type  x  y  z
                            f.write(f"\n{atom_type:4d}     {coord_x:8.5f}     "
                                   f"{coord_y:8.5f}   {coord_z:8.5f}")

        print(f"XYZ file saved to: {filename}")

    def print_statistics(self):
        """Print system statistics."""
        # Mapping of atom type names
        atom_type_names = {
            1: 'sub (bottom fixed layer)',
            2: 'water (bulk water phase)',
            3: 'gas (bubble region)',
            4: 'oil-head (hydrophilic end)',
            5: 'oil-tail (hydrophobic end)',
            6: 'top (top fixed layer)'
        }

        print("\n" + "="*70)
        print("System configuration statistics")
        print("="*70)
        print(f"Grid size: X={self.X}, Y={self.Y}, Z={self.Z}")
        print(f"Total grid points: {self.X * self.Y * self.Z}")
        print(f"Total atoms: {self.total_atoms}")
        print(f"Total bonds: {len(self.bonds)}")

        # Count atoms by type
        atom_type_count = {}
        for z in range(self.Z):
            for y in range(self.Y):
                for x in range(self.X):
                    if self.ntype[z][y][x] > 0:
                        atom_type = self.ntype[z][y][x]
                        atom_type_count[atom_type] = atom_type_count.get(atom_type, 0) + 1

        print("\nAtom count by type:")
        for atom_type in sorted(atom_type_count.keys()):
            count = atom_type_count[atom_type]
            percentage = (count / self.total_atoms) * 100
            type_name = atom_type_names.get(atom_type, 'Unknown type')
            print(f"  Type {atom_type} - {type_name:25s}: {count:6d} atoms ({percentage:5.2f}%)")
        print("="*70 + "\n")


def main():
    """
    Main entry point demonstrating how to use GridSystemGenerator.

    User-adjustable parameters:
    ---------------------------
    1. Grid dimensions: X_SIZE, Y_SIZE, Z_SIZE
    2. Top layer position: TOP_LAYER_OFFSET (controls the Z coordinate of the top fixed layer)
    3. Gas position and size: GAS_X_START, GAS_X_WIDTH
    4. Oil position and size: OIL_X_START, OIL_X_WIDTH
    5. Mixed phase depth: MIXED_PHASE_Z_START, MIXED_PHASE_Z_THICKNESS
    6. Output file names: LAMMPS_FILENAME, XYZ_FILENAME
    7. Number of atom types: NUM_ATOM_TYPES
    8. Region definitions: pass custom regions to define_regions()
    9. Bonds: pass custom bond_regions to create_bonds()
    """

    # ========== User input parameters ==========

    # Parameter 1: grid dimensions
    X_SIZE = 101  # Number of grid points along X
    Y_SIZE = 5    # Number of grid points along Y
    Z_SIZE = 81   # Number of grid points along Z

    # Parameter 2: top layer positioning
    TOP_LAYER_OFFSET = 7  # Downward offset from Z_SIZE for the top layer
                           # offset=10 -> top layer at z=69-70 when Z=81
                           # offset=5  -> top layer at z=74-75 when Z=81
                           # Recommended range: 5-20

    # Parameter 3: gas region position and size
    GAS_X_START = 21      # Starting X position of the gas region
    GAS_X_WIDTH = 20      # Gas width along X (default 10: x=21-30)

    # Parameter 4: oil region position and size
    OIL_X_START = 56      # Starting X position of the oil region
    OIL_X_WIDTH = 25      # Oil width along X (default 20: x=61-80)

    # Note: spacing between gas and oil = OIL_X_START - (GAS_X_START + GAS_X_WIDTH)
    #       Example: 61 - (21 + 10) = 30 (i.e., x=31-60 is water)

    # Parameter 5: mixed phase control along Z
    MIXED_PHASE_Z_START = 3       # Starting Z of the mixed phase
    MIXED_PHASE_Z_THICKNESS = 9   # Thickness of the mixed phase (default 6: z=3-8)

    # Parameter 6: output file names
    LAMMPS_FILENAME = "geo.dat"  # LAMMPS data file name
    XYZ_FILENAME = "geo.xyz"     # XYZ file name

    # Parameter 7: number of atom types
    NUM_ATOM_TYPES = 6

    # ====================================

    # Compute and display configuration information
    gas_x_end = GAS_X_START + GAS_X_WIDTH - 1
    oil_x_end = OIL_X_START + OIL_X_WIDTH - 1
    gas_oil_gap = OIL_X_START - gas_x_end - 1
    mixed_z_end = MIXED_PHASE_Z_START + MIXED_PHASE_Z_THICKNESS - 1

    print("Starting 3D grid generation...")
    print(f"Grid dimensions: X={X_SIZE}, Y={Y_SIZE}, Z={Z_SIZE}")
    print(f"\nPosition configuration:")
    print(f"  - Top fixed layer: z={Z_SIZE - TOP_LAYER_OFFSET - 2}-{Z_SIZE - TOP_LAYER_OFFSET - 1}")
    print(f"  - Mixed phase thickness: z={MIXED_PHASE_Z_START}-{mixed_z_end} (thickness={MIXED_PHASE_Z_THICKNESS})")
    print(f"  - Gas region: x={GAS_X_START}-{gas_x_end} (width={GAS_X_WIDTH})")
    print(f"  - Oil region: x={OIL_X_START}-{oil_x_end} (width={OIL_X_WIDTH})")
    print(f"  - Gas/oil spacing: {gas_oil_gap} (water region: x={gas_x_end+1}-{OIL_X_START-1})")

    # Create generator instance
    generator = GridSystemGenerator(
        x_size=X_SIZE, y_size=Y_SIZE, z_size=Z_SIZE,
        top_layer_offset=TOP_LAYER_OFFSET,
        gas_x_start=GAS_X_START, gas_x_width=GAS_X_WIDTH,
        oil_x_start=OIL_X_START, oil_x_width=OIL_X_WIDTH,
        mixed_phase_z_start=MIXED_PHASE_Z_START,
        mixed_phase_z_thickness=MIXED_PHASE_Z_THICKNESS
    )

    # Define regions (default configuration; custom regions can be supplied)
    print("\nDefining atom regions...")
    generator.define_regions()

    # Create bonds (using default configuration)
    print("Creating atomic bonds...")
    generator.create_bonds()

    # Print statistics
    generator.print_statistics()

    # Write output files
    print("Writing output files...")
    generator.write_lammps_data(filename=LAMMPS_FILENAME, num_atom_types=NUM_ATOM_TYPES)
    generator.write_xyz(filename=XYZ_FILENAME)

    print("\nAll files generated!")


if __name__ == "__main__":
    main()

