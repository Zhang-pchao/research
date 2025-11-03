import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Toggle whether to use the center of mass
use_com = False  # When False, compute mass density per atom

# File paths
topfile = '../mix.top'
grofile = '../eq.gro'
xtcfile = '../prod.xtc'

# Parse the .top file to obtain the number of molecules
def parse_top_molecules(topfile):
    molecules = {}
    with open(topfile, 'r') as f:
        lines = f.readlines()
        in_molecules_section = False
        for line in lines:
            if line.strip().startswith('[ molecules ]'):
                in_molecules_section = True
                continue
            if in_molecules_section:
                if line.strip().startswith(';') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) == 2:
                    compound, nmols = parts
                    molecules[compound] = int(nmols)
                else:
                    break
    return molecules

# Retrieve molecule counts
molecules = parse_top_molecules(topfile)
n_ethanol = molecules.get('ethanol', 0)
n_n2 = molecules.get('N2', 0)
n_water = molecules.get('SOL', 0)

# Create the Universe object
u = mda.Universe(grofile, xtcfile)

# Automatically select molecules
ethanol_atoms = None
if n_ethanol > 0:
    ethanol_atoms = u.select_atoms(f'resname MOL and resid 1:{n_ethanol}')
n2_atoms = u.select_atoms(f'resname MOL and resid {n_ethanol+1}:{n_ethanol+n_n2}')
water_atoms = u.select_atoms('resname SOL')

# Validate the number of selected atoms
if ethanol_atoms:
    print(f"Ethanol atoms: {len(ethanol_atoms)}")
print(f"N2 atoms: {len(n2_atoms)}")
print(f"Water atoms: {len(water_atoms)}")

# Define molecular masses (in atomic mass units)
ethanol_mass = 1.008*6+12.011*2+15.999
n2_mass = 14.007*2
water_mass = 1.008*2+15.999

# Density calculation settings
box = u.dimensions
z_max = box[2]
n_bins = 100
bin_edges = np.linspace(0, z_max, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Initialize density arrays
density_ethanol = np.zeros(n_bins)
density_n2 = np.zeros(n_bins)
density_water = np.zeros(n_bins)
density_total = np.zeros(n_bins)

# Atomic mass (amu) table (can be refined using the actual .top or .itp files)
mass_table = {
    'H': 1.008,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999
}

# Iterate over the trajectory
n_frames = 0
for ts in u.trajectory:
    if ts.frame < 100:
        continue

    if use_com:
        if n_ethanol > 0:
            ethanol_com = np.array([
                ethanol_atoms[i*9:(i+1)*9].center_of_mass()[2] for i in range(n_ethanol)
            ])
            ethanol_hist, _ = np.histogram(ethanol_com, bins=bin_edges)
            density_ethanol += ethanol_hist * ethanol_mass
        else:
            ethanol_hist = np.zeros(n_bins)

        n2_com = np.array([
            n2_atoms[i*2:(i+1)*2].center_of_mass()[2] for i in range(n_n2)
        ])
        water_com = np.array([
            water_atoms[i*3:(i+1)*3].center_of_mass()[2] for i in range(n_water)
        ])

        n2_hist, _ = np.histogram(n2_com, bins=bin_edges)
        water_hist, _ = np.histogram(water_com, bins=bin_edges)

        density_n2 += n2_hist * n2_mass
        density_water += water_hist * water_mass
        density_total += (ethanol_hist * ethanol_mass if n_ethanol > 0 else 0) + \
                         n2_hist * n2_mass + water_hist * water_mass

    else:
        # Atom-wise calculation
        def atomwise_density(atomgroup, label):
            hist = np.zeros(n_bins)
            for atom in atomgroup:
                z = atom.position[2]
                mass = atom.mass
                bin_idx = np.searchsorted(bin_edges, z) - 1
                if 0 <= bin_idx < n_bins:
                    hist[bin_idx] += mass
            return hist

        if n_ethanol > 0:
            ethanol_hist = atomwise_density(ethanol_atoms, 'ethanol')
            density_ethanol += ethanol_hist
        else:
            ethanol_hist = np.zeros(n_bins)

        n2_hist = atomwise_density(n2_atoms, 'n2')
        water_hist = atomwise_density(water_atoms, 'water')

        density_n2 += n2_hist
        density_water += water_hist
        density_total += ethanol_hist + n2_hist + water_hist

    n_frames += 1

if n_frames == 0:
    raise ValueError("No frames processed after skipping first frames.")

# Average over frames
density_ethanol /= n_frames
density_n2 /= n_frames
density_water /= n_frames
density_total /= n_frames

# Convert units to g/cm³
area = box[0] * box[1] * 1e-16  # Å² to cm²
bin_height = z_max / n_bins * 1e-8  # Å to cm
volume_per_bin = area * bin_height
avogadro = 6.02214076e23

density_ethanol = density_ethanol / avogadro / volume_per_bin * 1e3
density_n2 = density_n2 / avogadro / volume_per_bin * 1e3
density_water = density_water / avogadro / volume_per_bin * 1e3
density_total = density_total / avogadro / volume_per_bin * 1e3

# Save data
data = {
    'z (Å)': bin_centers,
    'Density_N2 (g/cm³)': density_n2,
    'Density_Water (g/cm³)': density_water,
    'Density_Total (g/cm³)': density_total
}
if n_ethanol > 0:
    data['Density_Ethanol (g/cm³)'] = density_ethanol

df = pd.DataFrame(data)
df.to_csv('density_data.csv', index=False)
print("Density data saved to 'density_data.csv'")

# Plotting
plt.figure(figsize=(10, 6))
if n_ethanol > 0:
    plt.plot(bin_centers, density_ethanol, label='Ethanol')
plt.plot(bin_centers, density_n2, label='Nitrogen')
plt.plot(bin_centers, density_water, label='Water')
plt.plot(bin_centers, density_total, label='Total')
plt.xlabel(r'z (Å)')
plt.ylabel(r'Density (g/cm$^3$)')
plt.legend()
plt.grid(True)
plt.savefig('density_plot.png')
plt.show()
print("Density plot saved to 'density_plot.png'")
