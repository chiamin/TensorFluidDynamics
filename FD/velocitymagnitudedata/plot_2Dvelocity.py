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
# Reshape to 2D grid
# -------------------------
Nx = len(np.unique(xs))
Ny = len(np.unique(ys))

fs_2d     = fs.reshape((Nx, Ny))
fs_qtt_2d = fs_qtt.reshape((Nx, Ny))

# Make coordinate matrices (for contour plotting)
X = xs.reshape((Nx, Ny))
Y = ys.reshape((Nx, Ny))

# -------------------------
# 1 Plot 2D original field
# -------------------------
plt.figure(figsize=(6,5))
plt.title("Original Interpolated Field")
plt.xlabel("x")
plt.ylabel("y")
plt.pcolormesh(X, Y, fs_2d, shading='auto')
plt.colorbar(label="Value")
plt.tight_layout()

# -------------------------
# 2 Plot 2D QTT field
# -------------------------
plt.figure(figsize=(6,5))
plt.title("Reconstructed Field (t=1e-3)")
plt.xlabel("x")
plt.ylabel("y")
plt.pcolormesh(X, Y, fs_qtt_2d, shading='auto')
plt.colorbar(label="Value")
plt.tight_layout()

# -------------------------
# 3 Plot bond dimensions
# -------------------------

plt.figure(figsize=(6,4))
plt.title("Fieled Bond Dimensions (t=1e-3)")
plt.plot(bonddims_v, marker='o')
plt.xlabel("Bond index") #bond index
plt.ylabel("Bond dimension D")
plt.grid(True)
plt.tight_layout()

plt.show()
