# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:46:29 2024

@author: Lab01
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tifffile as tiff

# Parameters
num_frames = 1000000  # choose number of frames to be analyzed

RJDP = np.zeros((20, 20), dtype=np.float64)
number_of_files = 1

for ii in range(number_of_files):
    file_path = f'gaussian_50percent_method_2_only_G_local_cube_{ii}.tif'  # Specify your TIFF file path here

    with tiff.TiffFile(file_path) as tif:
        total_frames = min(len(tif.pages), num_frames)

        for i in tqdm(range(total_frames - 2)):  # Loop over frames, ensuring enough frames for `frame2`
            frame1 = tif.pages[i].asarray().astype(np.float64)

            frame2 = tif.pages[i + 1].asarray().astype(np.float64)

            r1 = np.sum((frame1), axis=0)  # axis 0=x, axis 1=y
            c1 = np.transpose(r1)
            r2 = np.sum((frame2), axis=0)  # axis 0=x, axis 1=y

            S = np.outer(c1, r1)
            N = np.outer(c1, r2)

            JDP = S - N

            RJDP += JDP

# RJDP/1000000
# Set the zeroth diagonal elements to zero
np.fill_diagonal((RJDP), 0.0)

# Get the sums of all diagonals
diag_sums = [np.sum(np.diag((RJDP), k)) for k in
             range(-RJDP.shape[0] + 1, RJDP.shape[1])]  # np.rot90 (RJDP) momentum

# -------------------------
# Plot all results together
# -------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: RJDP ---
im0 = axs[0].imshow(np.rot90(RJDP), origin='lower')
axs[0].set_title('Nearfield y-axis JDP')
fig.colorbar(im0, ax=axs[1])

# --- Plot 2: Diagonal sums ---
axs[1].plot(diag_sums)
axs[1].set_title('Nearfield Diagonal Sums')
axs[1].set_xlabel('Diagonal index')
axs[1].set_ylabel('Sum')

# --- Plot 3: Single frame ---
im2 = axs[2].imshow(frame1, origin='lower')
axs[2].set_title('Single Frame S=3')
fig.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.show()
