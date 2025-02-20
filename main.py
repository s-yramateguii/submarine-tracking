import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
# utility for clearing output of cell as loop runs in notebook
from IPython.display import clear_output
from numpy.fft import fftn, ifftn, fftshift, ifftshift

'''
Averaging Fourier Transforms
'''
d = np.load("subdata.npy")
# NOTE: L we defined in class is 2Lh here, i.e. the domain here is [-Lh,Lh].
Lh = 10; # length of spatial domain (cube of side L = 2*10). 
N_grid = 64; # number of grid points/Fourier modes in each direction
xx = np.linspace(-Lh, Lh, N_grid+1) #spatial grid in x dir
x = xx[0:N_grid]
y = x # same grid in y,z direction
z = x

K_grid = (2*np.pi/(2*Lh))*np.linspace(-N_grid/2, N_grid/2 -1, N_grid) # frequency grid for one coordinate

xv, yv, zv = np.meshgrid(x, y, z) # generate 3D meshgrid for plotting

KX, KY, KZ = np.meshgrid(K_grid, K_grid, K_grid)
aver = np.zeros((N_grid, N_grid, N_grid), dtype=complex)

for j in range(49):
    sig = np.reshape(d[:, j], (N_grid, N_grid, N_grid))
    aver += fftn(sig)

aver_shift = np.abs(fftshift(aver)/49)
aver_max = np.max(aver_shift)
max_index = np.unravel_index(np.argmax(aver_shift), aver_shift.shape)

# Determine dominant frequency
dfreq = (KX[max_index], KY[max_index], KZ[max_index])
print(aver_max)
print("Dominant frequency: " + str(dfreq))

# Plots showing dominant frequency in the KX-KZ and KX-KY planes
y_plot = max_index[0]
z_plot = max_index[2]
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))

fig.tight_layout(pad=3.0)

# Plot for KX-KZ plane
contour0 = ax0.contourf(KX[y_plot, :, :], KZ[:, y_plot, :], np.abs(aver_shift[y_plot, :, :]), 
                        cmap='viridis')
fig.colorbar(contour0, ax=ax0, label='Amplitude')
ax0.scatter(dfreq[0], dfreq[2], color='red', s=100, label='Dominant Frequency')
ax0.set_title("Dominant Frequency KX-KZ Plane")
ax0.set_xlabel("KX")
ax0.set_ylabel("KZ")
ax0.legend()

# Plot for KX-KY plane
contour1 = ax1.contourf(KX[:, :, z_plot], KY[:, :, z_plot], np.abs(aver_shift[:, :, z_plot]), 
                        cmap='viridis')
fig.colorbar(contour1, ax=ax1, label='Amplitude')
ax1.scatter(dfreq[0], dfreq[1], color='red', s=100, label='Dominant Frequency')
ax1.set_title("Dominant Frequency KX-KY Plane")
ax1.set_xlabel("KX")
ax1.set_ylabel("KY")
ax1.legend()
# plt.savefig(f"dominant_frequency.png", dpi=300, bbox_inches='tight')
plt.show()

'''
Filtering
'''
# Apply the filter to the data
sigma = 3
filter = np.exp(-1/(2 * sigma**2)*((KX - dfreq[0])**2 + (KY - dfreq[1])**2 + (KZ - dfreq[2])**2))

# Filtered Coordinates
xc = np.zeros(49)
yc = np.zeros(49)
zc = np.zeros(49)

# Noisy Coordinates
xn = np.zeros(49)
yn = np.zeros(49)
zn = np.zeros(49)

for j in range(49):
    sig = np.reshape(d[:, j], (N_grid, N_grid, N_grid))
    sig_fft = fftshift(fftn(sig))
    sig_filter = sig_fft * filter  # Apply the filter
    sigd = np.real(ifftn(ifftshift(sig_filter)))
    sign = np.real(ifftn(ifftshift(sig_fft))) # No filter applied
    max_index_filter = np.unravel_index(np.argmax(sigd), sigd.shape)
    max_index_noise = np.unravel_index(np.argmax(sign), sign.shape)

    xc[j] = xv[max_index_filter]
    yc[j] = yv[max_index_filter]
    zc[j] = zv[max_index_filter]

    xn[j] = xv[max_index_noise]
    yn[j] = yv[max_index_noise]
    zn[j] = zv[max_index_noise]


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xn, yn, zn, color='r', label='Noisy Path', marker='o')
ax.plot(xc, yc, zc, color='b', label='Denoised Path', marker='o')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Noisy vs Denoised Submarine Path')
ax.legend()
# plt.savefig(f"denoise.png", dpi=300, bbox_inches='tight')
plt.show()

'''
Plotting xy Coordinates of Submarine Path
'''
plt.figure(figsize=(10, 8))
plt.plot(xc, yc, marker='o', linestyle='-', color='#1f77b4', markersize=8, linewidth=2, label="Submarine Path")
plt.scatter(xc[0], yc[0], color='green', s=100, label="Start", zorder=5)  # Start point
plt.scatter(xc[-1], yc[-1], color='red', s=100, label="End", zorder=5)    # End point
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('Path of Submarine', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.legend(fontsize=12)
# plt.savefig(f"sub_path.png", dpi=300, bbox_inches='tight')
plt.show()

# plot iso surfaces for every third measurement

# for j in range(0,49,3):

#   signal = np.reshape(d[:, j], (N_grid, N_grid, N_grid))
#   normal_sig_abs = np.abs(signal)/np.abs(signal).max()

#   # generate data for isosurface of the 3D data 
#   fig_data = go.Isosurface( x = xv.flatten(), y = yv.flatten(), z = zv.flatten(),
#                            value = normal_sig_abs.flatten(), isomin=0.6, isomax=1)

#   # generate plots
#   clear_output(wait=True) # need this to discard previous figs
#   fig = go.Figure( data = fig_data )
#   fig.show()
