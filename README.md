# Tensor Fluid Dynamics - TCI Comparison Analysis

This project uses Tensor Cross Interpolation (TCI) methods to compress and analyze fluid dynamics data, comparing the performance of original TCI and adaptive patched TCI.

## Project Structure

```
.
├── FD/                          # Fluid Dynamics section
│   ├── velocitymagnitudedata/   # Velocity magnitude data analysis
│   └── T4AAdaptivePatchedTCI.jl/ # TCI library
├── GPE/                         # Gross-Pitaevskii Equation section
│   └── vortex_func/             # Vortex function analysis
└── ref/                         # References
```

## Requirements

### Julia Environment
- Julia 1.10 or higher
- Required packages will be automatically installed by the scripts

### Python Environment
- Python 3.7+
- Install packages: `pip install numpy matplotlib h5py`

## Usage

### Part 1: FD (Fluid Dynamics) - Velocity Magnitude Data Analysis

#### 1. Prepare Data

Ensure you have velocity data files in HDF5 format (e.g., `f_t1e-8.h5`) containing:
- `xs`: x coordinate array
- `ys`: y coordinate array
- `fs`: velocity magnitude array

#### 2. Run TCI Analysis

```bash
cd FD/velocitymagnitudedata
julia apply_adaptive_tci.jl
```

**Key Parameters (modify in `apply_adaptive_tci.jl`):**
```julia
h5file = "f_t1e-8.h5"           # Input data file
R = 8                            # Grid resolution (2^R × 2^R)
tol = 1e-7                       # TCI tolerance
maxbonddim = 500                 # Original TCI max bond dimension
maxbonddim_patched = 60           # Patched TCI max bond dimension
ordering = :interleaved          # :fused_xy, :fused_yx, or :interleaved
```

#### 3. Visualize Results

```bash
python plot_errors_from_file.py
```

This generates:
- `f_t1e-8_errors_2d_maps.png`: 2D error heatmaps
- `f_t1e-8_errors_data_with_patches.png`: Data plot with patch boundaries
- `f_t1e-8_errors_statistics.png`: Error statistics plot

---

### Part 2: GPE (Gross-Pitaevskii Equation) - Vortex Function Analysis

#### 1. Prepare Data

Ensure you have HDF5 files containing vortex states (e.g., `125v.h5`).

**Data File Format:**

HDF5 files should contain one of the following:

**Format 1: Pre-evaluated Grid Data**
```
125v.h5
├── xs: float array    # x coordinates
├── ys: float array    # y coordinates
├── fs: float array    # vortex function values
└── metadata/          # metadata (optional)
```

**Format 2: MPS Tensor Format**
- Contains MPS tensor data
- Scripts will automatically convert to interpolation functions

**Creating Interpolator (if needed):**
```bash
julia read_vortex_h5.jl
```

#### 2. Run TCI Analysis

```bash
cd GPE/vortex_func
julia apply_adaptive_tci_vortex.jl
```

**Key Parameters (modify in `apply_adaptive_tci_vortex.jl`):**
```julia
# Data range
xmin = -21.0
xmax = 21.0
ymin = -21.0
ymax = 21.0

# TCI parameters
R = 8                    # Grid resolution
tol = 1e-7              # Tolerance
maxbonddim = 300        # Original TCI max bond dimension
maxbonddim_patched = 60  # Patched TCI max bond dimension
ordering = :fused        # Ordering scheme
fused_order = :xy       # Fused ordering sequence
```

#### 3. Visualize Results

```bash
python plot_errors_from_file.py
```

The script reads the error file directly (filename is hardcoded in the script).

---

## Output Data Format

Both sections generate error files in HDF5 format:

```
*_errors.h5
├── xs, ys, fs                    # Coordinates and true values
├── errors_original_tci           # Original TCI errors
├── errors_patched_tci            # Patched TCI errors
├── time_original_tci             # Original TCI computation time
├── time_patched_tci              # Patched TCI computation time
├── bonddims_original_tci         # Original TCI bond dimensions
├── bonddims_patched_tci          # Patched TCI bond dimensions
├── patch_bounds                  # Patch boundaries (optional)
└── metadata/                     # Tolerance, maxbonddim, R, npoints
```
