# -*- coding: utf-8 -*-
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sys

# Suppress specific MDAnalysis warnings if needed (optional)
warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis.analysis.hydrogenbonds')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='MDAnalysis.coordinates.XTC')

# --- Configuration ---
# File paths
topfile = '../mix.top'  # Topology file (GROMACS .top)
grofile = '../eq.gro'  # Coordinate file (GROMACS .gro)
xtcfile = '../prod.xtc' # Trajectory file (GROMACS .xtc)

# Analysis Parameters
skip_frames = 100 # <<< Number of initial frames to skip
z_min = 0      # Minimum Z value for histogram (Angstrom) - Adjust based on your box
z_max = 159.33 # Maximum Z value for histogram (Angstrom) - Adjust based on your box
n_bins_z = 101   # Number of bins along the Z-axis
d_a_cutoff = 3.5 # Donor-Acceptor distance cutoff (Angstrom)
d_h_a_angle_cutoff = 140 # Donor-Hydrogen-Acceptor angle cutoff (degrees)

# --- Molecule Naming Conventions (CRITICAL: Match your files) ---
ethanol_top_name = 'ethanol'
water_top_name = 'SOL'
ethanol_resname = 'MOL' # As seen in eq.gro
water_resname = 'SOL'   # As seen in eq.gro
ethanol_O_name = 'O8'    # Oxygen in ethanol's -OH group (donor/acceptor)
ethanol_H_name = 'H9'    # Hydrogen in ethanol's -OH group (donor H)
water_O_name = 'OW'     # Oxygen in water (donor/acceptor)
water_H_names = ['HW1', 'HW2'] # Hydrogens in water (donor H)

# Output Files
output_csv_file = 'hbond_z_distribution_local_norm.csv' # Changed name
output_plot_file = 'hbond_z_distribution_local_norm.png' # Changed name
output_ethanol_csv_file = 'ethanol_total_hbond_z_distribution.csv' # Ethanol-specific output
output_ethanol_plot_file = 'ethanol_total_hbond_z_distribution.png' # Ethanol-specific plot
# --- End Configuration ---

# --- Functions ---
def parse_top_molecules(topfile):
    """Parses a GROMACS-style .top file to get molecule counts."""
    molecules = {}
    try:
        with open(topfile, 'r') as f:
            lines = f.readlines()
            in_molecules_section = False
            for line in lines:
                line = line.strip()
                if not line or line.startswith(';'): continue
                if line.startswith('[ molecules ]'):
                    in_molecules_section = True
                    continue
                if in_molecules_section:
                    if line.startswith('['): break
                    parts = line.split()
                    if len(parts) >= 2:
                        compound = parts[0]
                        try:
                            nmols = int(parts[-1])
                            molecules[compound] = nmols
                        except ValueError:
                            print(f"Warning: Could not parse number of molecules for line: {line}")
    except FileNotFoundError:
        print(f"Error: Topology file not found at {topfile}")
        return None
    except Exception as e:
        print(f"Error reading topology file {topfile}: {e}")
        return None
    if not molecules: print("Warning: No molecules found in the [ molecules ] section.")
    return molecules

# --- Main Script ---
print("Starting Hydrogen Bond Analysis...")

# 1. Load System
print(f"Loading universe from {grofile} and {xtcfile}...")
try:
    u = mda.Universe(grofile, xtcfile)
except Exception as e:
    print(f"Error creating MDAnalysis Universe: {e}"); sys.exit(1)

total_frames = u.trajectory.n_frames
print(f"Universe loaded with {u.atoms.n_atoms} atoms and {total_frames} frames.")

if skip_frames >= total_frames:
    print(f"Error: skip_frames ({skip_frames}) >= total_frames ({total_frames})."); sys.exit(1)
if skip_frames < 0:
    print("Error: skip_frames cannot be negative."); sys.exit(1)
if skip_frames > 0: print(f"Skipping the first {skip_frames} frames.")

n_frames_analyzed = total_frames - skip_frames
if n_frames_analyzed <= 0:
    print(f"Error: No frames to analyze ({n_frames_analyzed}). Check skip_frames."); sys.exit(1)
print(f"Analyzing {n_frames_analyzed} frames (frame index {skip_frames} to {total_frames - 1}).")

# 2. Get Molecule Counts
print(f"Parsing molecule counts from {topfile}...")
molecule_counts = parse_top_molecules(topfile)
if molecule_counts is None: sys.exit(1)

n_ethanol = molecule_counts.get(ethanol_top_name, 0)
n_water = molecule_counts.get(water_top_name, 0)
print(f"Found {n_ethanol} '{ethanol_top_name}' (resname {ethanol_resname}).")
print(f"Found {n_water} '{water_top_name}' (resname {water_resname}).")

if n_ethanol == 0 and n_water == 0:
    print("Error: No ethanol or water molecules found."); sys.exit(1)

# 3. Define Selections
donors_sel_list, hydrogens_sel_list, acceptors_sel_list = [], [], []
donor_molecule_defs = {} # Store selection for donor atoms of each type

if n_ethanol > 0:
    eth_donor_sel = f"(resname {ethanol_resname} and name {ethanol_O_name})"
    donors_sel_list.append(eth_donor_sel)
    hydrogens_sel_list.append(f"(resname {ethanol_resname} and name {ethanol_H_name})")
    acceptors_sel_list.append(eth_donor_sel)
    donor_molecule_defs[ethanol_resname] = eth_donor_sel
if n_water > 0:
    wat_donor_sel = f"(resname {water_resname} and name {water_O_name})"
    donors_sel_list.append(wat_donor_sel)
    if water_H_names:
        water_h_sel = " or ".join([f"name {hname}" for hname in water_H_names])
        hydrogens_sel_list.append(f"(resname {water_resname} and ({water_h_sel}))")
    acceptors_sel_list.append(wat_donor_sel)
    donor_molecule_defs[water_resname] = wat_donor_sel

donors_sel = " or ".join(donors_sel_list) if donors_sel_list else "none"
hydrogens_sel = " or ".join(hydrogens_sel_list) if hydrogens_sel_list else "none"
acceptors_sel = " or ".join(acceptors_sel_list) if acceptors_sel_list else "none"

print("\n--- Selections ---")
print(f"Donors:    {donors_sel}")
print(f"Hydrogens: {hydrogens_sel}")
print(f"Acceptors: {acceptors_sel}")
for res, sel in donor_molecule_defs.items(): print(f"Donor Atoms ({res}): {sel}")
print("------------------\n")

if donors_sel == "none" or hydrogens_sel == "none" or acceptors_sel == "none":
    print("Error: One or more required selections are empty ('none')."); sys.exit(1)

# Verify selections in the first frame
try:
    ts_check = u.trajectory[skip_frames]
    for name, sel in [("Donors", donors_sel), ("Hydrogens", hydrogens_sel), ("Acceptors", acceptors_sel)]:
        n_atoms = len(u.select_atoms(sel))
        print(f"Frame {skip_frames}: Found {n_atoms} potential {name.lower()} atoms.")
        if n_atoms == 0: print(f"Warning: Selection '{sel}' found 0 atoms. Check naming conventions!")
    for res, sel in donor_molecule_defs.items():
         n_atoms = len(u.select_atoms(sel))
         print(f"Frame {skip_frames}: Found {n_atoms} potential donor-defining atoms for {res} ('{sel}').")
         if n_atoms == 0: print(f"Warning: Selection for {res} donor atoms is empty!")

except IndexError: print(f"Error: Cannot access frame {skip_frames}."); sys.exit(1)
except mda.SelectionError as e: print(f"Error verifying selections: {e}"); sys.exit(1)

# 4. Run Hydrogen Bond Analysis
print("\nInitializing HydrogenBondAnalysis...")
print(f"Using D-A cutoff: {d_a_cutoff} Å, D-H-A angle cutoff: {d_h_a_angle_cutoff} deg")
try:
    hbonds = HydrogenBondAnalysis(
        universe=u, donors_sel=donors_sel, hydrogens_sel=hydrogens_sel,
        acceptors_sel=acceptors_sel, d_a_cutoff=d_a_cutoff,
        d_h_a_angle_cutoff=d_h_a_angle_cutoff, update_selections=False
    )
    print(f"Running H-bond analysis over {n_frames_analyzed} frames...")
    hbonds.run(start=skip_frames, verbose=False) # verbose=True for detailed progress
except Exception as e: print(f"Error during H-bond analysis: {e}"); sys.exit(1)

n_total_hbonds_found = len(hbonds.results.hbonds)
print(f"Analysis complete. Found {n_total_hbonds_found} total H-bond instances.")

# 5. Process Results - Calculate Z-distributions with LOCAL Normalization
if n_total_hbonds_found == 0:
    print("Warning: No H-bonds found. Cannot calculate distributions."); sys.exit(0)

# Extract H-bond data: [frame, donor_idx, acceptor_idx]
hbond_data = hbonds.results.hbonds[:, [0, 1, 3]].astype(int)

print("Preparing atom maps and initializing counters...")
# Map atom index to (resname, array_index)
atom_indices = u.atoms.indices
atom_resnames = u.atoms.resnames
atom_map = {idx: (resname, i) for i, (idx, resname) in enumerate(zip(atom_indices, atom_resnames))}

# Z-bin setup
z_bins = np.linspace(z_min, z_max, n_bins_z)
z_centers = (z_bins[:-1] + z_bins[1:]) / 2.0

# Initialize dictionaries for counts
# Raw H-bond counts per bin (binned by DONOR Z)
raw_hbond_counts_per_bin = {}
# Raw donor molecule counts per bin (using donor defining atom)
donor_molecule_counts_per_bin = {}
# Overall total H-bond counts (for average calculation later)
hbond_total_counts = {}

possible_hb_types = []
if n_ethanol > 0: possible_hb_types.append(f"{ethanol_resname}-{ethanol_resname}")
if n_ethanol > 0 and n_water > 0: possible_hb_types.append(f"{ethanol_resname}-{water_resname}")
if n_water > 0 and n_ethanol > 0: possible_hb_types.append(f"{water_resname}-{ethanol_resname}")
if n_water > 0: possible_hb_types.append(f"{water_resname}-{water_resname}")

for hb_type in possible_hb_types:
    raw_hbond_counts_per_bin[hb_type] = np.zeros(len(z_centers), dtype=np.int64)
    hbond_total_counts[hb_type] = 0

# Initialize donor molecule counts per bin
for resname in donor_molecule_defs.keys():
    donor_molecule_counts_per_bin[resname] = np.zeros(len(z_centers), dtype=np.int64)

# Initialize ethanol-involving H-bond counts per bin (binned by ETHANOL Z-coordinate)
# This will include both ethanol as donor and ethanol as acceptor
ethanol_hbond_counts_per_bin = np.zeros(len(z_centers), dtype=np.int64)

# Create AtomGroups for donor defining atoms
donor_atom_groups = {}
try:
    for resname, sel in donor_molecule_defs.items():
        donor_atom_groups[resname] = u.select_atoms(sel)
        if not donor_atom_groups[resname]:
             print(f"Warning: AtomGroup for {resname} donor atoms is empty!")
except mda.SelectionError as e:
     print(f"Error creating donor atom groups: {e}"); sys.exit(1)


# Iterate through trajectory ONCE to get Z positions and calculate distributions
print(f"Processing {n_frames_analyzed} frames for Z-distributions and counts...")
# Map absolute frame index to list index (0 to n_frames_analyzed-1)
frame_idx_map = {frame_i: list_idx for list_idx, frame_i in enumerate(range(skip_frames, total_frames))}
# Filter hbond_data to only include analyzed frames (should be done by hbonds.run(start=...) already, but safer)
analyzed_frame_indices = set(range(skip_frames, total_frames))
hbond_data_filtered = hbond_data[np.isin(hbond_data[:, 0], list(analyzed_frame_indices))]
del hbond_data # Free memory

# Store Z positions temporarily frame by frame to avoid huge memory usage if possible
# Or pre-collect all Zs if memory allows (as before)
print("Collecting Z-positions and calculating distributions frame-by-frame...")
frames_processed = 0
for i in range(skip_frames, total_frames):
    ts = u.trajectory[i]
    current_zpos = ts.positions[:, 2]
    frame_idx = ts.frame # Absolute frame index

    # --- A: Calculate Donor Molecule Distribution for this frame ---
    for resname, ag in donor_atom_groups.items():
        if ag: # Check if group exists and is not empty
            donor_z_coords = current_zpos[ag.indices] # Get Z for these specific atoms
            # Calculate histogram for THIS FRAME
            hist_donors, _ = np.histogram(donor_z_coords, bins=z_bins)
            # Accumulate counts over frames
            donor_molecule_counts_per_bin[resname] += hist_donors

    # --- B: Process H-bonds found in this frame ---
    # Find rows in hbond_data_filtered corresponding to the current frame_idx
    frame_hbond_indices = np.where(hbond_data_filtered[:, 0] == frame_idx)[0]

    for row_idx in frame_hbond_indices:
        _, donor_idx, acceptor_idx = hbond_data_filtered[row_idx]

        try:
            donor_resname, donor_atom_array_idx = atom_map[donor_idx]
            acceptor_resname, _ = atom_map[acceptor_idx] # Acceptor resname needed for type

            # --- Determine H-bond type ---
            hbond_type = f"{donor_resname}-{acceptor_resname}"

            if hbond_type in raw_hbond_counts_per_bin: # Check if we track this type
                 # --- Get DONOR Z coordinate ---
                 z_donor = current_zpos[donor_atom_array_idx]

                 # --- Find Z-bin index for the DONOR ---
                 bin_idx = np.digitize(z_donor, z_bins) - 1

                 # --- Increment counts (if bin is valid) ---
                 if 0 <= bin_idx < len(z_centers):
                      raw_hbond_counts_per_bin[hbond_type][bin_idx] += 1
                 #else: Optional: count bonds outside the defined z_min/z_max range

                 # Increment overall total count for this H-bond type
                 hbond_total_counts[hbond_type] += 1

                 # --- Additional tracking for ethanol-involving H-bonds (binned by ETHANOL Z) ---
                 # Each H-bond counts once per ethanol molecule involved:
                 # - For ethanol-ethanol bonds: counts once for donor's z-bin AND once for acceptor's z-bin
                 # - For ethanol-water bonds: counts once for the ethanol's z-bin (as donor)
                 # - For water-ethanol bonds: counts once for the ethanol's z-bin (as acceptor)
                 # This gives us the average number of H-bonds per ethanol molecule at each z position.
                 if n_ethanol > 0:
                     if donor_resname == ethanol_resname:
                         # Ethanol as donor: use donor's Z (already calculated)
                         if 0 <= bin_idx < len(z_centers):
                             ethanol_hbond_counts_per_bin[bin_idx] += 1
                     if acceptor_resname == ethanol_resname:
                         # Ethanol as acceptor: use acceptor's Z
                         # Changed from 'elif' to 'if' so ethanol-ethanol bonds count for BOTH molecules
                         acceptor_resname_full, acceptor_atom_array_idx = atom_map[acceptor_idx]
                         z_acceptor = current_zpos[acceptor_atom_array_idx]
                         acceptor_bin_idx = np.digitize(z_acceptor, z_bins) - 1
                         if 0 <= acceptor_bin_idx < len(z_centers):
                             ethanol_hbond_counts_per_bin[acceptor_bin_idx] += 1

        except KeyError as e:
             # Simplified error handling
             print(f"Warning: Frame {frame_idx}, Atom index {e} not found in map. Skipping H-bond.")
             continue # Skip this H-bond
        except IndexError as e:
             print(f"Warning: Frame {frame_idx}, IndexError accessing Z-position. Error: {e}. Skipping H-bond.")
             continue # Skip this H-bond

    frames_processed += 1
    if frames_processed % 100 == 0 or frames_processed == n_frames_analyzed:
         print(f"  Processed {frames_processed}/{n_frames_analyzed} frames...")

print("Distribution calculation complete.")
del hbond_data_filtered # Free memory

# 6. Calculate Final Normalized Distributions and Overall Averages

print("\nCalculating final locally normalized Z-distributions...")
final_hbond_distribution = {}
# Define mapping from H-bond type to the corresponding donor residue name
donor_type_map = {}
if n_ethanol > 0: donor_type_map[f"{ethanol_resname}-{ethanol_resname}"] = ethanol_resname
if n_ethanol > 0 and n_water > 0: donor_type_map[f"{ethanol_resname}-{water_resname}"] = ethanol_resname
if n_water > 0 and n_ethanol > 0: donor_type_map[f"{water_resname}-{ethanol_resname}"] = water_resname
if n_water > 0: donor_type_map[f"{water_resname}-{water_resname}"] = water_resname

for hb_type in possible_hb_types:
    # Total raw H-bonds of this type in each bin (summed over frames)
    total_hb_in_bin = raw_hbond_counts_per_bin[hb_type]

    # Get the corresponding donor type (e.g., 'MOL' or 'SOL')
    donor_resname = donor_type_map.get(hb_type)
    if donor_resname is None:
        print(f"Warning: Cannot find donor type for H-bond type {hb_type}. Skipping distribution.")
        final_hbond_distribution[hb_type] = np.zeros_like(z_centers)
        continue

    # Total raw donor molecules of the corresponding type found in each bin (summed over frames)
    total_donors_in_bin = donor_molecule_counts_per_bin.get(donor_resname)
    if total_donors_in_bin is None:
         print(f"Warning: Cannot find donor counts for donor type {donor_resname}. Skipping distribution for {hb_type}.")
         final_hbond_distribution[hb_type] = np.zeros_like(z_centers)
         continue

    # --- Perform the LOCAL normalization ---
    # Avg H-bonds per donor molecule IN THAT BIN
    # Use np.divide for safe division by zero
    final_hbond_distribution[hb_type] = np.divide(
        total_hb_in_bin,
        total_donors_in_bin,
        out=np.zeros_like(total_hb_in_bin, dtype=float), # Output is zero if denominator is zero
        where=total_donors_in_bin > 0 # Condition for division
    )

# --- Calculate Overall Average (normalized by TOTAL donor molecules in box) ---
# This calculation remains the same as before, for comparison/context
print("\n--- Average Hydrogen Bond Counts per Molecule per Frame (Overall Box) ---")
print(f"(Normalization: D-A bonds are normalized by the TOTAL number of D molecules in the box)")
print(f"(Calculated as: Total D-A bonds found / (Num Frames * Total Num D Molecules))")
print("-----------------------------------------------------------------------")
overall_averages = {}
# Use total molecule counts (n_ethanol, n_water) for this overall average
overall_norm_factors = {}
if n_ethanol > 0: overall_norm_factors[ethanol_resname] = n_ethanol
if n_water > 0: overall_norm_factors[water_resname] = n_water

for hb_type in possible_hb_types:
    total_count = hbond_total_counts.get(hb_type, 0)
    donor_resname = donor_type_map.get(hb_type)
    norm_factor = overall_norm_factors.get(donor_resname, 0)

    if n_frames_analyzed > 0 and norm_factor > 0 and total_count > 0:
        avg_per_mol_per_frame = total_count / (n_frames_analyzed * norm_factor)
        print(f"{hb_type}: {avg_per_mol_per_frame:.4f}")
        overall_averages[hb_type] = avg_per_mol_per_frame
    else:
        print(f"{hb_type}: 0.0000")
        overall_averages[hb_type] = 0.0

# Special reporting for water coordination number estimate (based on overall average)
ww_hb_type = f"{water_resname}-{water_resname}"
if ww_hb_type in overall_averages:
    avg_ww_per_mol_overall = overall_averages[ww_hb_type]
    estimated_coord_num = avg_ww_per_mol_overall * 2
    print("-" * 20)
    print(f"Total {ww_hb_type} bonds found: {hbond_total_counts.get(ww_hb_type, 0)}")
    print(f"Avg {ww_hb_type} bonds per water molecule (Overall Box Avg): {avg_ww_per_mol_overall:.4f}")
    print(f"Estimated average H-bond coordination number for water: {estimated_coord_num:.4f}")
print("-----------------------------------------------------------------------\n")

# --- Calculate Additional Distributions ---
# 6.1. Calculate per-ethanol-molecule hydrogen bond distribution
print("Calculating per-ethanol-molecule H-bond Z-distribution...")

# Use the complete ethanol-involving H-bond data (includes ethanol as both donor and acceptor)
# This includes: ethanol-ethanol, ethanol-water (ethanol as donor), and water-ethanol (ethanol as acceptor)
if n_ethanol > 0:
    # Normalize by ethanol molecule count in each bin
    ethanol_donors_per_bin = donor_molecule_counts_per_bin.get(ethanol_resname, np.zeros(len(z_centers)))
    ethanol_hbonds_per_molecule = np.divide(
        ethanol_hbond_counts_per_bin,  # This now includes ALL ethanol-involving H-bonds
        ethanol_donors_per_bin,
        out=np.zeros_like(ethanol_hbond_counts_per_bin, dtype=float),
        where=ethanol_donors_per_bin > 0
    )
else:
    ethanol_hbonds_per_molecule = np.zeros(len(z_centers))

# 6.2. Calculate per-all-molecule hydrogen bond distribution
print("Calculating per-all-molecule H-bond Z-distribution...")
all_total_hbonds_per_bin = np.zeros(len(z_centers), dtype=np.int64)
all_total_molecules_per_bin = np.zeros(len(z_centers), dtype=np.int64)

# Sum all H-bonds (from all donor types)
for hb_type in possible_hb_types:
    all_total_hbonds_per_bin += raw_hbond_counts_per_bin[hb_type]

# Sum all donor molecules (ethanol + water)
for resname in donor_molecule_defs.keys():
    all_total_molecules_per_bin += donor_molecule_counts_per_bin[resname]

# Normalize by total molecule count in each bin
all_hbonds_per_molecule = np.divide(
    all_total_hbonds_per_bin,
    all_total_molecules_per_bin,
    out=np.zeros_like(all_total_hbonds_per_bin, dtype=float),
    where=all_total_molecules_per_bin > 0
)

# 7. Save Z-Distribution Data (Locally Normalized) to CSV
print(f"Saving LOCALLY normalized Z-distribution data to {output_csv_file}...")
df_out = pd.DataFrame({'Z_center_A': z_centers})
for hb_type, distribution in final_hbond_distribution.items():
    # Column name reflects the local normalization
    col_name = f'{hb_type}_per_donor_in_bin'
    df_out[col_name] = distribution

# Add the new columns for additional distributions
df_out['ethanol_all_HB_per_ethanol_molecule'] = ethanol_hbonds_per_molecule  # Includes ethanol as donor & acceptor
df_out['all_total_HB_per_all_molecule'] = all_hbonds_per_molecule

try:
    df_out.to_csv(output_csv_file, index=False, float_format='%.6e')
    print(f"CSV file saved successfully to {output_csv_file}")
except Exception as e:
    print(f"Error saving CSV file: {e}")

# 8. Plot Z-Distribution (Locally Normalized)
print(f"Generating plot {output_plot_file}...")
plt.figure(figsize=(12, 7))

plot_labels = {
    f"{ethanol_resname}-{ethanol_resname}": f"{ethanol_resname}(D)-{ethanol_resname}(A)",
    f"{ethanol_resname}-{water_resname}": f"{ethanol_resname}(D)-{water_resname}(A)",
    f"{water_resname}-{ethanol_resname}": f"{water_resname}(D)-{ethanol_resname}(A)",
    f"{water_resname}-{water_resname}": f"{water_resname}(D)-{water_resname}(A)",
}

plotted_something = False
for hb_type, distribution in final_hbond_distribution.items():
    if np.any(distribution > 0):
        label = plot_labels.get(hb_type, hb_type)
        plt.plot(z_centers, distribution, label=label, linewidth=1.5)
        plotted_something = True

# Add the new distribution curves
if n_ethanol > 0 and np.any(ethanol_hbonds_per_molecule > 0):
    plt.plot(z_centers, ethanol_hbonds_per_molecule, label=f'{ethanol_resname} All HB/molecule (donor+acceptor)', 
             linewidth=2.0, linestyle='--', color='red')
    plotted_something = True

if np.any(all_hbonds_per_molecule > 0):
    plt.plot(z_centers, all_hbonds_per_molecule, label='All Total HB/molecule', 
             linewidth=2.0, linestyle='-.', color='black')
    plotted_something = True

if not plotted_something:
     print("Warning: No locally normalized H-bond data with non-zero values to plot.")

plt.xlabel('Z-coordinate of Donor Atom (Å)')
# Y-axis label reflects the local normalization
plt.ylabel('Avg. H-Bonds / Donor Molecule (in Z-bin)')
plt.title(f'Locally Normalized H-Bond Distribution along Z (Frames {skip_frames}-{total_frames-1})\n'
          f'Cutoffs: D-A ≤ {d_a_cutoff} Å, D-H-A ≥ {d_h_a_angle_cutoff}°')
plt.xlim(z_min, z_max)
plt.ylim(bottom=0)
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

if plotted_something:
    plt.legend(fontsize='medium', loc='best')

plt.tight_layout()

try:
    plt.savefig(output_plot_file, dpi=300)
    print(f"Plot saved to {output_plot_file}")
except Exception as e:
    print(f"Error saving plot: {e}")

# plt.show()

# 9. Create separate output for ethanol total H-bond distribution
if n_ethanol > 0:
    print(f"\nGenerating separate ethanol H-bond distribution output...")
    
    # Prepare ethanol-specific DataFrame
    df_ethanol = pd.DataFrame({'Z_center_A': z_centers})
    df_ethanol['ethanol_total_HB_per_molecule'] = ethanol_hbonds_per_molecule
    
    # Also include individual ethanol-involved bond types for reference
    if f"{ethanol_resname}-{ethanol_resname}" in final_hbond_distribution:
        df_ethanol[f'{ethanol_resname}-{ethanol_resname}_per_donor'] = final_hbond_distribution[f"{ethanol_resname}-{ethanol_resname}"]
    if f"{ethanol_resname}-{water_resname}" in final_hbond_distribution:
        df_ethanol[f'{ethanol_resname}-{water_resname}_per_donor'] = final_hbond_distribution[f"{ethanol_resname}-{water_resname}"]
    if f"{water_resname}-{ethanol_resname}" in final_hbond_distribution:
        df_ethanol[f'{water_resname}-{ethanol_resname}_per_donor'] = final_hbond_distribution[f"{water_resname}-{ethanol_resname}"]
    
    # Add ethanol molecule count per bin for reference
    ethanol_donors_per_bin = donor_molecule_counts_per_bin.get(ethanol_resname, np.zeros(len(z_centers)))
    df_ethanol['ethanol_molecules_per_bin_total'] = ethanol_donors_per_bin / n_frames_analyzed
    
    # Save ethanol-specific CSV
    try:
        df_ethanol.to_csv(output_ethanol_csv_file, index=False, float_format='%.6e')
        print(f"Ethanol-specific CSV saved to {output_ethanol_csv_file}")
    except Exception as e:
        print(f"Error saving ethanol CSV file: {e}")
    
    # Create ethanol-specific plot
    print(f"Generating ethanol-specific plot {output_ethanol_plot_file}...")
    plt.figure(figsize=(12, 7))
    
    # Main curve: total ethanol H-bonds per molecule
    if np.any(ethanol_hbonds_per_molecule > 0):
        plt.plot(z_centers, ethanol_hbonds_per_molecule, 
                label=f'{ethanol_resname} Total HB/molecule (as Donor + Acceptor)', 
                linewidth=2.5, color='red', marker='o', markersize=3, markevery=5)
    
    # Add breakdown by bond type (optional, for detailed view)
    if f"{ethanol_resname}-{ethanol_resname}" in final_hbond_distribution:
        plt.plot(z_centers, final_hbond_distribution[f"{ethanol_resname}-{ethanol_resname}"],
                label=f'{ethanol_resname}(D)-{ethanol_resname}(A)', 
                linewidth=1.5, linestyle='--', alpha=0.7)
    if f"{ethanol_resname}-{water_resname}" in final_hbond_distribution:
        plt.plot(z_centers, final_hbond_distribution[f"{ethanol_resname}-{water_resname}"],
                label=f'{ethanol_resname}(D)-{water_resname}(A)', 
                linewidth=1.5, linestyle='--', alpha=0.7)
    if f"{water_resname}-{ethanol_resname}" in final_hbond_distribution:
        plt.plot(z_centers, final_hbond_distribution[f"{water_resname}-{ethanol_resname}"],
                label=f'{water_resname}(D)-{ethanol_resname}(A)', 
                linewidth=1.5, linestyle='--', alpha=0.7)
    
    plt.xlabel('Z-coordinate (Å)', fontsize=12)
    plt.ylabel('Avg. H-Bonds per Ethanol Molecule', fontsize=12)
    plt.title(f'Ethanol Total H-Bond Distribution along Z\n'
              f'(Includes ethanol as both Donor and Acceptor, Frames {skip_frames}-{total_frames-1})\n'
              f'Cutoffs: D-A ≤ {d_a_cutoff} Å, D-H-A ≥ {d_h_a_angle_cutoff}°', fontsize=11)
    plt.xlim(z_min, z_max)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize='medium', loc='best')
    plt.tight_layout()
    
    try:
        plt.savefig(output_ethanol_plot_file, dpi=300)
        print(f"Ethanol-specific plot saved to {output_ethanol_plot_file}")
    except Exception as e:
        print(f"Error saving ethanol plot: {e}")
else:
    print("\nNo ethanol molecules found, skipping ethanol-specific output.")

print("\nAnalysis finished.")