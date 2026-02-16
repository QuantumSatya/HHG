# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tifffile as tiff

# Parameters
num_frames = 500000
number_of_files = 1

r0, c0 = 25, 15   # reference pixel

# Will initialize after reading first frame
G_corr = None          # Σ (n0*n - n0*n_shift)
S_sum  = None          # Σ n(x,y)
S0_sum = 0.0           # Σ n0
M = 0                  # number of used pairs

for ii in range(number_of_files):
    file_path = f'NF_GaussianEnvelope_SPDC_{ii}.tif'

    with tiff.TiffFile(file_path) as tif:
        total_frames = min(len(tif.pages), num_frames)

        # initialize arrays from frame shape
        if G_corr is None:
            H, W = tif.pages[0].asarray().shape
            G_corr = np.zeros((H, W), dtype=np.float64)
            S_sum  = np.zeros((H, W), dtype=np.float64)

        for i in tqdm(range(total_frames - 1), desc=f"File {ii}"):
            frame1 = tif.pages[i].asarray().astype(np.float64)
            frame2 = tif.pages[i + 1].asarray().astype(np.float64)

            n0 = frame1[r0, c0]   # scalar

            # accumulate singles
            S0_sum += n0
            S_sum  += frame1

            # accumulate correlated numerator (same - accidentals)
            G_corr += n0 * (frame1 - frame2)

            M += 1

# Normalize to g2 map: g2(x,y) = (M * G_corr(x,y)) / (S0_sum * S_sum(x,y))
den = S0_sum * S_sum
g2 = np.full_like(G_corr, np.nan, dtype=np.float64)
valid = den > 0
g2[valid] = (M * G_corr[valid]) / den[valid]

# Plot
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# --- g2 map ---
im0 = axs[0].imshow(g2, origin='lower')
axs[0].set_title(rf'$g^{(2)}$ map vs ref pixel ({r0},{c0})')
fig.colorbar(im0, ax=axs[0])

# --- single frame ---
im1 = axs[1].imshow(frame2, origin='lower')
axs[1].set_title('Single frame (example)')
fig.colorbar(im1, ax=axs[1])

# --- line slice (proper way: use plot, not imshow) ---
axs[2].plot(g2[r0, :])
axs[2].set_title(f'Line slice at row r0 = {r0}')
axs[2].set_xlabel('Column index')
axs[2].set_ylabel('g2')
axs[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
