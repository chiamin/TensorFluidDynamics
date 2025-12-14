import h5py
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load from HDF5
# -------------------------

with h5py.File("f_t1e-3.h5", "r") as f:
    xs       = f["xs"][:]        
    ys       = f["ys"][:]        
    fs       = f["fs"][:]        
    fs_qtt   = f["fs_qtt"][:]    
    bonddims_v = f["bonddims"][:]  

# -------------------------
# Determine grid dimensions
# -------------------------
Nx = len(np.unique(xs))
Ny = len(np.unique(ys))

# Get unique coordinate values (sorted)
x_coords = np.sort(np.unique(xs))
y_coords = np.sort(np.unique(ys))

# Number of bits needed for each dimension
nx_bits = int(np.log2(Nx))  # 8 bits for 256 points
ny_bits = int(np.log2(Ny))  # 8 bits for 256 points
total_bits = nx_bits + ny_bits  # 16 bits total

print(f"Grid dimensions: {Nx} x {Ny}")
print(f"Bits: {nx_bits} for x, {ny_bits} for y, {total_bits} total")

# -------------------------
# Create lookup dictionary for faster access
# -------------------------
# Create a 2D lookup: (x_idx, y_idx) -> f(x,y)
fs_2d = fs.reshape((Nx, Ny))
fs_qtt_2d = fs_qtt.reshape((Nx, Ny))

# Create coordinate index mapping
x_to_idx = {x_val: idx for idx, x_val in enumerate(x_coords)}
y_to_idx = {y_val: idx for idx, y_val in enumerate(y_coords)}


# -------------------------
# Function: Binary number -> (x, y) coordinates
# -------------------------
def binary_to_coords(binary_num, x_coords=x_coords, y_coords=y_coords, 
                     nx_bits=nx_bits, ny_bits=ny_bits):
    """
    Maps a binary number to (x, y) coordinates using interleaved bit pattern.
    
    The binary representation is: (b_x^1, b_y^1, b_x^2, b_y^2, ...)
    where b_x^1 is the smallest scale (LSB) for x.
    
    Parameters:
    -----------
    binary_num : int
        Binary number representing the coordinate
    x_coords : array
        Sorted array of unique x coordinates
    y_coords : array
        Sorted array of unique y coordinates
    nx_bits : int
        Number of bits for x coordinate
    ny_bits : int
        Number of bits for y coordinate
    
    Returns:
    --------
    (x, y) : tuple
        The (x, y) coordinate pair
    """
    # Extract interleaved bits
    x_bits = []
    y_bits = []
    
    # Extract bits in interleaved order: x[0], y[0], x[1], y[1], ...
    for i in range(max(nx_bits, ny_bits)):
        if i < nx_bits:
            # x bit at position 2*i (0, 2, 4, ...)
            x_bit = (binary_num >> (2*i)) & 1
            x_bits.append(x_bit)
        if i < ny_bits:
            # y bit at position 2*i+1 (1, 3, 5, ...)
            y_bit = (binary_num >> (2*i + 1)) & 1
            y_bits.append(y_bit)
    
    # Convert bit arrays to indices
    x_idx = sum(bit * (2**i) for i, bit in enumerate(x_bits))
    y_idx = sum(bit * (2**i) for i, bit in enumerate(y_bits))
    
    # Map indices to actual coordinates
    x = x_coords[x_idx]
    y = y_coords[y_idx]
    
    return (x, y)


# -------------------------
# Function: Binary number -> f(x, y)
# -------------------------
def f_from_binary(binary_num, field_data=fs_2d, x_coords=x_coords, y_coords=y_coords,
                  nx_bits=nx_bits, ny_bits=ny_bits):
    """
    Maps a binary number to the function value f(x, y).
    
    Parameters:
    -----------
    binary_num : int
        Binary number representing the coordinate
    field_data : 2D array
        The field data array (Nx x Ny)
    x_coords : array
        Sorted array of unique x coordinates
    y_coords : array
        Sorted array of unique y coordinates
    nx_bits : int
        Number of bits for x coordinate
    ny_bits : int
        Number of bits for y coordinate
    
    Returns:
    --------
    f_value : float
        The function value at (x, y)
    """
    # Get coordinates from binary number
    x, y = binary_to_coords(binary_num, x_coords, y_coords, nx_bits, ny_bits)
    
    # Get indices
    x_idx = x_to_idx[x]
    y_idx = y_to_idx[y]
    
    # Return function value
    return field_data[x_idx, y_idx]


# -------------------------
# Verify the mapping by plotting
# -------------------------

# Create coordinate grids for plotting
# We need to create a grid where each position corresponds to a binary number
# with interleaved bits

# First, create a mapping from (x_idx, y_idx) to binary number
def coords_to_binary(x_idx, y_idx, nx_bits=nx_bits, ny_bits=ny_bits):
    """
    Maps (x_idx, y_idx) to a binary number using interleaved bits.
    Inverse of binary_to_coords.
    """
    binary_num = 0
    # Interleave bits: x[0], y[0], x[1], y[1], ...
    for i in range(max(nx_bits, ny_bits)):
        if i < nx_bits:
            x_bit = (x_idx >> i) & 1
            binary_num |= (x_bit << (2*i))
        if i < ny_bits:
            y_bit = (y_idx >> i) & 1
            binary_num |= (y_bit << (2*i + 1))
    return binary_num

# Create a grid using binary mapping
# For each binary number, get (x,y) and then look up f(x,y)
fs_reconstructed_flat = np.zeros(Nx * Ny)
x_reconstructed = np.zeros(Nx * Ny)
y_reconstructed = np.zeros(Nx * Ny)

for binary_num in range(Nx * Ny):
    x, y = binary_to_coords(binary_num, x_coords, y_coords, nx_bits, ny_bits)
    x_reconstructed[binary_num] = x
    y_reconstructed[binary_num] = y
    fs_reconstructed_flat[binary_num] = f_from_binary(binary_num, fs_2d, x_coords, y_coords, nx_bits, ny_bits)

# Reshape to 2D for plotting
fs_reconstructed = fs_reconstructed_flat.reshape((Nx, Ny))
X_reconstructed = x_reconstructed.reshape((Nx, Ny))
Y_reconstructed = y_reconstructed.reshape((Nx, Ny))

# Reshape original data for comparison
fs_2d_original = fs.reshape((Nx, Ny))
X = xs.reshape((Nx, Ny))
Y = ys.reshape((Nx, Ny))

# -------------------------
# Plot comparison
# -------------------------

# 1. Original field
plt.figure(figsize=(6, 5))
plt.title("Original Field (from HDF5)")
plt.xlabel("x")
plt.ylabel("y")
plt.pcolormesh(X, Y, fs_2d_original, shading='auto')
plt.colorbar(label="Value")
plt.tight_layout()

# 2. Verify the function works correctly
# We need to compare values at the same (x,y) coordinates
# Create a lookup for original data: (x, y) -> f(x,y)
original_lookup = {}
for i in range(len(xs)):
    original_lookup[(xs[i], ys[i])] = fs[i]

# Compare by checking if same (x,y) gives same f(x,y)
matches = 0
mismatches = []
for binary_num in range(Nx * Ny):
    x, y = binary_to_coords(binary_num, x_coords, y_coords, nx_bits, ny_bits)
    f_reconstructed = f_from_binary(binary_num, fs_2d, x_coords, y_coords, nx_bits, ny_bits)
    if (x, y) in original_lookup:
        f_original = original_lookup[(x, y)]
        if abs(f_reconstructed - f_original) < 1e-10:
            matches += 1
        else:
            mismatches.append((x, y, f_original, f_reconstructed))

print(f"\nVerification:")
print(f"Total points: {Nx * Ny}")
print(f"Matches: {matches}")
print(f"Mismatches: {len(mismatches)}")
if len(mismatches) > 0:
    print(f"First few mismatches:")
    for i, (x, y, f_orig, f_recon) in enumerate(mismatches[:5]):
        print(f"  ({x:.6f}, {y:.6f}): original={f_orig:.6f}, reconstructed={f_recon:.6f}, diff={abs(f_orig-f_recon):.6e}")

# 3. Plot side-by-side comparison
# For proper visualization, we'll create sorted grids
# Create a sorted version for the reconstructed data
x_sorted = np.sort(x_coords)
y_sorted = np.sort(y_coords)
fs_sorted = np.zeros((Nx, Ny))

# Create a mapping from (x, y) to sorted indices
x_sorted_to_idx = {x_val: idx for idx, x_val in enumerate(x_sorted)}
y_sorted_to_idx = {y_val: idx for idx, y_val in enumerate(y_sorted)}

# Fill the sorted grid by iterating through all binary numbers
for binary_num in range(Nx * Ny):
    x, y = binary_to_coords(binary_num, x_coords, y_coords, nx_bits, ny_bits)
    f_val = f_from_binary(binary_num, fs_2d, x_coords, y_coords, nx_bits, ny_bits)
    x_sorted_idx = x_sorted_to_idx[x]
    y_sorted_idx = y_sorted_to_idx[y]
    fs_sorted[x_sorted_idx, y_sorted_idx] = f_val

# Create coordinate matrices for sorted data
X_sorted, Y_sorted = np.meshgrid(x_sorted, y_sorted, indexing='ij')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original (using original grid structure)
ax = axes[0]
ax.set_title("Original Field (from HDF5)")
ax.set_xlabel("x")
ax.set_ylabel("y")
im1 = ax.pcolormesh(X, Y, fs_2d_original, shading='auto')
plt.colorbar(im1, ax=ax, label="Value")

# Reconstructed (sorted for proper visualization)
ax = axes[1]
ax.set_title("Reconstructed Field (from Binary Mapping, sorted)")
ax.set_xlabel("x")
ax.set_ylabel("y")
im2 = ax.pcolormesh(X_sorted, Y_sorted, fs_sorted, shading='auto')
plt.colorbar(im2, ax=ax, label="Value")

plt.tight_layout()

if matches == Nx * Ny:
    print("\n✓ Results match perfectly! The binary mapping function works correctly.")
else:
    print(f"\n⚠ Warning: {len(mismatches)} points don't match. This might be due to coordinate ordering differences.")

plt.show()

