import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, MaxNLocator
import matplotlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NumpyTensorTools')))
import qtt_tools as qtt
import npmps
import plot_utility as pltut
import plotsetting as ps

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# Configuration
STEPS = 125  # Number of vortices (modify as needed)
KILL_SITES = 9  # Number of sites to kill for downsampling
X_RANGE = (-21, 21)  # Coordinate range

input_mps_path = f"{STEPS}v.pkl"
with open(input_mps_path, 'rb') as file:
    data = pickle.load(file)
mps = data

x1, x2 = X_RANGE
# Kill sites for downsampling
for i in range(KILL_SITES):
    mps = qtt.kill_site_2D(mps, 80, dtype=np.complex128)

mps = qtt.normalize_MPS_by_integral (mps, x1, x2, Dim=2)
N = len(mps)//2
Ndx = 2**N
rescale = (x2-x1)/Ndx
shift = x1
print(npmps.MPS_dims(mps))
bxs = list(pltut.BinaryNumbers(N))
bys = list(pltut.BinaryNumbers(N))

xs = pltut.bin_to_dec_list (bxs, rescale, shift)
ys = pltut.bin_to_dec_list (bys, rescale, shift)
X, Y = np.meshgrid (xs, ys)


Z = pltut.get_2D_mesh_eles_mps(mps, bxs, bys)
# Save raw data to file
X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
np.savetxt("psi2_2D.dat", np.column_stack((X_flat, Y_flat, Z_flat)),
           fmt="%.8e", delimiter="\t", header="x\ty\t|psi|^2")
Z = np.abs(Z)**2
fig, ax = plt.subplots()
ax.relim()
ax.autoscale_view()
ax.tick_params(axis='both', which='major', labelsize=20)
ax.plot(X[Ndx//2,:],Z[Ndx//2,:], label=r'$| \psi |^{2}$', color='black')
ax.legend(loc='upper right', fontsize=14)
#ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MultipleLocator(0.01))
ax.set_xlabel(r'$x$', fontsize=20)
ax.set_ylabel(r'$| \mathrm{\psi} |^{2}$', fontsize=20)
ps.text(ax, x=0.1, y=0.9, t="(a)", fontsize=20)
ax.set_xlim(x1, x2)
plt.savefig("2D_psi2_sec.pdf", transparent=False)



fig2,ax2 = plt.subplots()
ax2.relim()
ax2.autoscale_view()
ax2.tick_params(axis='both', which='major', labelsize=10)
surfxy = ax2.contourf(X, Y, Z, cmap=cm.coolwarm, antialiased=False)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.set_xlabel(r'$x$', rotation=0, fontsize=20)
ax2.set_ylabel(r'$y$', rotation=0, fontsize=20)
ax2.set_aspect('equal', adjustable='box')
cbar = fig2.colorbar(surfxy)
cbar.ax.tick_params(labelsize=20)
ps.set_tick_inteval(ax2.yaxis, major_itv=5, minor_itv=1)
ps.set_tick_inteval(ax2.xaxis, major_itv=5, minor_itv=1)
ax2.set_ylim(x1, x2)
ax2.set_xlim(x1, x2)
ps.text(ax2, x=0.1, y=0.9, t="(b)", fontsize=20)
plt.savefig(f"{input_mps_path}.pdf", bbox_inches='tight')

