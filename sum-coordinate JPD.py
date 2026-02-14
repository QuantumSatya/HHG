# -*- coding: utf-8 -*-
"""
Sum-coordinate projection JPD computation and export
"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tifffile as tiff

# Parameters
num_frames = 100000000
number_of_files = 1

# File path template  (FIXED)
file_template = "Masked_NF_GaussianEnvelope_SPDC_{}.tif"

# Initialize on first frame (FIXED)
with tiff.TiffFile(file_template.format(0)) as tif:
    n = tif.pages[0].asarray().shape[0]
    signal = np.zeros((2 * n - 1, 2 * n - 1), dtype=np.float64)
    noise = np.zeros_like(signal)
    count1 = np.zeros_like(signal)
    count2 = np.zeros_like(signal)

# Loop over all files
for ii in range(number_of_files):
    file_path = file_template.format(ii)  # FIXED
    with tiff.TiffFile(file_path) as tif:
        total_frames = min(len(tif.pages), num_frames)
        for f in tqdm(range(total_frames - 1), desc=f"Processing file {ii+1}/{number_of_files}"):
            frame1 = np.rot90(tif.pages[f].asarray().astype(np.float64))
            frame2 = np.rot90(tif.pages[f + 1].asarray().astype(np.float64))

            for i in range(n):
                for j in range(n):
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            i2 = i + di
                            j2 = j + dj
                            if 0 <= i2 < n and 0 <= j2 < n:
                                bi = 2 * i + di
                                bj = 2 * j + dj
                                signal[bi, bj] += frame1[i, j] * frame1[i2, j2]
                                noise[bi, bj] += frame1[i, j] * frame2[i2, j2]
                                count1[bi, bj] += 1
                                count2[bi, bj] += 1

# Final averaged maps
with np.errstate(divide='ignore', invalid='ignore'):
    Rsignal = np.where(count1 != 0, signal / count1, 0)
    Rnoise = np.where(count2 != 0, noise / count2, 0)

# Subtract noise
sumJPD = Rsignal - Rnoise

# Normalize by diagonal sum per offset
normalized_sumJPD = np.zeros_like(sumJPD)
for di in range(-n + 1, n):
    for dj in range(-n + 1, n):
        indices_i, indices_j = [], []
        for i in range(n):
            for j in range(n):
                bi, bj = 2 * i + di, 2 * j + dj
                if 0 <= bi < 2 * n - 1 and 0 <= bj < 2 * n - 1:
                    indices_i.append(bi)
                    indices_j.append(bj)
        values = sumJPD[indices_i, indices_j]
        diag_sum = np.sum(values)
        if diag_sum != 0:
            normalized_sumJPD[indices_i, indices_j] = values / diag_sum

# Save result
np.save("sumJPD.npy", normalized_sumJPD)

# Plot
plt.figure(figsize=(6, 6), dpi=300)
plt.imshow(normalized_sumJPD, cmap='inferno', vmin=0, vmax=normalized_sumJPD.max())
plt.xticks([])
plt.yticks([])
plt.tight_layout(pad=0)
plt.savefig("sumJPD_plot.png", dpi=600, bbox_inches='tight')
plt.show()
