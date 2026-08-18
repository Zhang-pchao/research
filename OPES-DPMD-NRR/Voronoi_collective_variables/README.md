# `VORONOID3` for OPES-DPMD proton-transfer environments

This directory contains [`VoronoiD3.cpp`](./VoronoiD3.cpp), a PLUMED custom collective variable used by the OPES-DPMD nitrogen-reduction workflow. It measures a defect-weighted distance between ordinary water O centers and one or more special reactive sites, such as the top N atom of an N₂H-related environment.

The file is a **project-specific legacy Action**. For a general implementation with explicit selections and current `VORONOI_COORDINATION`, `VORONOI_DISTANCE`, and `VORONOI_POSITION` Actions, start from the [Reactive Soft-Voronoi collective-variable guide](https://zhang-pchao.github.io/code/reactive-voronoi). The guide also explains how to validate exact and neighbor-list modes before adding OPES bias.

## Action and geometry

The source registers:

```text
VORONOID3
```

Its legacy input model is:

- `GROUPA`: ordinary water O centers followed by the special reactive centers.
- `GROUPB`: transferable H atoms.
- `NRX`: number of special centers at the end of `GROUPA`.
- `D_0`: reference occupancy for ordinary water O centers.
- `D_1`, `D_2`, `D_3`: reference occupancies for the first three special centers.
- `LAMBDA`: distance-kernel parameter used by the source as `exp(LAMBDA * distance)`.

For the standard NRR input, `GROUPA=WaterO,N2topN` and `NRX=1`; the code therefore treats the last `GROUPA` entry as the special N center. The reduction includes only pairs between ordinary water O sites and the special sites. It does not automatically include every pair inside `GROUPA`.

The resulting scalar is a signed defect-weighted distance, not a bare geometric distance and not a formal charge. Its interpretation depends on the reference occupancies, the ordering of `GROUPA`, and the sign/magnitude of `LAMBDA`.

## Build a runtime plugin

Compile with the exact PLUMED executable and ABI used by the LAMMPS/DeePMD/OPES job:

```bash
cd /path/to/research/OPES-DPMD-NRR/Voronoi_collective_variables
mkdir -p build-voronoi-d3
cd build-voronoi-d3
plumed mklib \
  ../VoronoiD3.cpp
```

The command produces a shared library whose name depends on the PLUMED version, normally `VoronoiD3.so`. Rebuild it whenever PLUMED, the compiler, MPI, or the ABI changes. Runtime compilation keeps the custom Action isolated from the main PLUMED installation; copying the source into `plumed/src/colvar` and rebuilding PLUMED is the alternative integrated-installation route.

## Minimal input

```plumed
LOAD FILE=./VoronoiD3.so
UNITS LENGTH=A

WaterO:  GROUP ATOMS=...
WaterH:  GROUP ATOMS=...
ReactiveN: GROUP ATOMS=...

# GROUPA order is significant: ordinary water O first, reactive sites last.
Centers: GROUP ATOMS=WaterO,ReactiveN

d3: VORONOID3 GROUPA=Centers GROUPB=WaterH NRX=1 LAMBDA=-5 \
    D_0=2 D_1=1

PRINT ARG=d3 FILE=COLVAR STRIDE=1
DUMPDERIVATIVES ARG=d3 FILE=DERIVATIVES STRIDE=1
```

The repository's production-style example is [`Enhanced_Sampling/OPES_MD/input.plumed`](../Enhanced_Sampling/OPES_MD/input.plumed). Its relevant setup is:

```plumed
WaterO:  GROUP ATOMS=108-160:1
WaterH:  GROUP ATOMS=1-107:1
N2topN:  GROUP ATOMS=166

d3: VORONOID3 GROUPA=WaterO,N2topN GROUPB=WaterH \
    NRX=1 LAMBDA=-5 D_0=2 D_1=1 \
    NLIST NL_CUTOFF=3.0 NL_STRIDE=1
```

The atom numbers are specific to that coordinate/topology ordering. Do not copy them to a new NRR or solvent model without checking the data file and the `GROUPA` order.

## Neighbor list

Without `NLIST`, the Action is the full-pair reference. With `NLIST`, both `NL_CUTOFF` and `NL_STRIDE` are required:

```plumed
d3_exact: VORONOID3 GROUPA=Centers GROUPB=WaterH NRX=1 \
    LAMBDA=-5 D_0=2 D_1=1

d3_trial: VORONOID3 GROUPA=Centers GROUPB=WaterH NRX=1 \
    LAMBDA=-5 D_0=2 D_1=1 \
    NLIST NL_CUTOFF=3.0 NL_STRIDE=1
```

The archived `NL_CUTOFF=3.0` is a setting for the NRR test system, not a transferable distance. This legacy source has no `NL_SKIN` keyword. Before production:

1. Compare exact and NLIST values on reactant, proton-transfer, product, host-switching, and distorted frames.
2. Compare analytical coordinate/box derivatives as well as scalar values.
3. Increase `NL_CUTOFF` until the required tolerance is met, then test `NL_STRIDE>1`.
4. Confirm that every transferable H retains at least one candidate center throughout the tested frames.
5. Only after these checks add `OPES_METAD_EXPLORE`, walls, restarts, and multi-walker settings.

`NL_STRIDE` must also be compatible with any replica-exchange or synchronized exchange schedule. Report the exact baseline, cutoff, stride, comparison tolerance, and the source/plugin checksum with the production record.

## OPES integration

The archived workflow biases `ss0` and `d1`; `d3` is one of the related distance coordinates used for comparison. A minimal bias block is shown below only to illustrate placement:

```plumed
opes: OPES_METAD_EXPLORE ...
  LABEL=opes
  ARG=ss0,d3
  FILE=HILLS
  TEMP=300
  PACE=100
  SIGMA_MIN=0.01,0.01
  BARRIER=55
...
```

The `TEMP`, `PACE`, `SIGMA_MIN`, `BARRIER`, walls, restart policy, and walker settings are sampling choices for the archived NRR workflow. They are not defaults of `VORONOID3` and should not be transferred without a new free-energy and force-validation plan.

## Validation and reproducibility record

Use a short unbiased `plumed driver` or equivalent fixed-frame test before coupling to LAMMPS:

```bash
plumed driver --plumed plumed.dat --ixyz representative.xyz
```

Record at minimum:

- repository and source commit;
- PLUMED version and `plumed mklib` command;
- compiler, MPI, accelerator/MD engine, and ABI;
- coordinate/topology atom order and the expanded `GROUPA`, `GROUPB` selections;
- `NRX`, `D_0...D_3`, `LAMBDA`, and NLIST settings;
- exact-versus-NLIST value and derivative errors;
- plugin checksum and the final PLUMED input.

A successful driver run confirms that the Action parses and evaluates on the supplied frames. It does not establish that a bias is physically appropriate, that the forces are correct for all relevant geometries, or that an OPES run has converged.

## Migration path

The current tutorial provides a safer general mapping:

| Legacy concept | Current API |
| --- | --- |
| Smooth occupancy defects | `VORONOI_COORDINATION` with explicit `CENTERS`, `ASSIGNED`, and `REFERENCE` |
| Water--reactive-site defect distance | `VORONOI_DISTANCE` with explicit `GROUP1` and `GROUP2` |
| Index or slab-position moments | `VORONOI_POSITION` with a physical `AXIS` and fixed `ORIGIN` |

The current API avoids inferring chemistry from the last `NRX` entries and makes the reference occupancy and pair semantics explicit. Keep `VoronoiD3.cpp` for reproducing the published OPES-DPMD NRR workflow; use the [complete guide](https://zhang-pchao.github.io/code/reactive-voronoi) for new systems.

## References

- [OPES-DPMD-NRR project](../README.md)
- [Archived OPES input](../Enhanced_Sampling/OPES_MD/input.plumed)
- [Source: `VoronoiD3.cpp`](./VoronoiD3.cpp)
- [Research repository](https://github.com/Zhang-pchao/research)
- [Reactive Soft-Voronoi guide](https://zhang-pchao.github.io/code/reactive-voronoi)
