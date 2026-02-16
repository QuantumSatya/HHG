# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tifffile as tiff

# Parameters
num_frames = 100000
number_of_files = 1

r0, c0 = 25, 25   # reference pixel

# Initialize after reading first frame
G_same = None      # Σ (n0 * n) same-frame
G_acc  = None      # Σ (n0 * n_shift) accidental (next frame)
S_sum  = None      # Σ n(x,y) singles map
S0_sum = 0.0       # Σ n0 singles scalar
M = 0              # number of used pairs

for ii in range(number_of_files):
    file_path = f'NF_GaussianEnvelope_SPDC_{ii}.tif'

    with tiff.TiffFile(file_path) as tif:
        total_frames = min(len(tif.pages), num_frames)

        if G_same is None:
            H, W = tif.pages[0].asarray().shape
            G_same = np.zeros((H, W), dtype=np.float64)
            G_acc  = np.zeros((H, W), dtype=np.float64)
            S_sum  = np.zeros((H, W), dtype=np.float64)

        for i in tqdm(range(total_frames - 1), desc=f"File {ii}"):
            frame1 = tif.pages[i].asarray().astype(np.float64)
            frame2 = tif.pages[i + 1].asarray().astype(np.float64)

            n0 = frame1[r0, c0]  # scalar reference-pixel intensity in THIS frame

            # accumulate singles
            S0_sum += n0
            S_sum  += frame1

            # accumulate coincidences
            G_same += n0 * frame1     # same-frame
            G_acc  += n0 * frame2     # accidentals (next frame)

            M += 1

# -----------------------------
# Normalize
# -----------------------------
den = S0_sum * S_sum

g2_std = np.full_like(G_same, np.nan, dtype=np.float64)     # baseline ~ 1
g2_excess = np.full_like(G_same, np.nan, dtype=np.float64)  # baseline ~ 0 (can be negative)
g2_std_minus1 = np.full_like(G_same, np.nan, dtype=np.float64)

valid = den > 0
g2_std[valid] = (M * G_same[valid]) / den[valid]
g2_excess[valid] = (M * (G_same[valid] - G_acc[valid])) / den[valid]
g2_std_minus1[valid] = g2_std[valid] - 1.0

# Optional: mask low-singles regions to avoid noisy edges
# (Recommended for sparse data)
thr = 1e-3 * np.nanmax(S_sum)  # tune: 1e-4 ... 1e-2 depending on sparsity
mask = S_sum > thr
g2_std[~mask] = np.nan
g2_excess[~mask] = np.nan
g2_std_minus1[~mask] = np.nan

# -----------------------------
# Diagnostics: neighbor correlation
# -----------------------------
# g2 between (r0,c0) and (r0,c0+1)
if c0 + 1 < W:
    print("g2_std(ref -> right neighbor) =", g2_std[r0, c0 + 1])
    print("g2_excess(ref -> right neighbor) =", g2_excess[r0, c0 + 1])
    print("g2_std_minus1(ref -> right neighbor) =", g2_std_minus1[r0, c0 + 1])

# -----------------------------
# Plot
# -----------------------------
fig, axs = plt.subplots(1, 4, figsize=(24, 5))

# 1) Standard g2 map (baseline ~1)
im0 = axs[0].imshow(g2_std, origin='lower')
axs[0].set_title(rf'Standard $g^{(2)}$ vs ref ({r0},{c0}) (baseline~1)')
fig.colorbar(im0, ax=axs[0])

# 2) Excess map (same - accidentals) (baseline ~0)
im1 = axs[1].imshow(g2_excess, origin='lower')
axs[1].set_title(rf'Excess corr: $(\langle nn\rangle-\langle nn\rangle_{{acc}})$ normalized')
fig.colorbar(im1, ax=axs[1])

# 3) Single frame
im2 = axs[2].imshow(frame1, origin='lower')
axs[2].set_title('Single frame (example)')
fig.colorbar(im2, ax=axs[2])

# 4) Line slice: show standard g2 and excess (optional) on same axis
axs[3].plot(g2_std[r0, :], label=r'$g^{(2)}_{\rm std}$')
axs[3].plot(g2_excess[r0, :], label=r'$g^{(2)}_{\rm excess}$', alpha=0.8)
axs[3].axvline(c0, linestyle='--', linewidth=1)
axs[3].set_title(f'Row slice at r0 = {r0}')
axs[3].set_xlabel('Column index')
axs[3].set_ylabel('value')
axs[3].grid(alpha=0.3)
axs[3].legend()

plt.tight_layout()
plt.show()
