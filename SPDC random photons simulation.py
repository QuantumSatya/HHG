#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate N random 8-bit HxW images.
In each image: `num_pairs` nearest-neighbor pixel pairs (horizontal or vertical)
have equal intensity. Pair base locations are drawn from a Gaussian envelope
centered in the image.

IMPORTANT (as requested):
- Only pixels that belong to correlated pairs are non-zero.
- All other pixels are exactly zero.
- No pixel is reused across pairs (no overlaps). Each image has `2*num_pairs`
  distinct non-zero pixels (unless impossible due to grid size).

@author: satyajeet
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from tqdm import tqdm

# -----------------------------
# Parameters
# -----------------------------
num_matrices = 200000
H, W = 20, 20

num_pairs = 100                 # number of correlated nearest-neighbor pairs per image
sigma = 8.0                     # Gaussian envelope width (pixels)
pair_value_mode = "random"      # "random" or "fixed"
fixed_pair_value = 250          # used only if pair_value_mode == "fixed"

out_path = "NF_GaussianEnvelope_SPDC_0.tif"
# -----------------------------

# Basic sanity check: cannot have64 more pairs than half the pixels (if no overlap)
max_pairs_possible = (H * W) // 2
if num_pairs > max_pairs_possible:
    raise ValueError(
        f"num_pairs={num_pairs} too large for {H}x{W} without overlap. "
        f"Max possible is {max_pairs_possible}."
    )

# Center of Gaussian
cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

# Gaussian weights over pixels
yy, xx = np.mgrid[0:H, 0:W]
gauss = np.exp(-(((yy - cy) ** 2) + ((xx - cx) ** 2)) / (2.0 * sigma ** 2))
weights = gauss.ravel()
weights /= weights.sum()

# Neighbor offsets: right and down
orientations = np.array([[0, 1], [1, 0]], dtype=np.int32)  # (dy, dx)

# Write frames one-by-one (memory-safe)
with tiff.TiffWriter(out_path, bigtiff=True) as tif:
    first_img = None

    for i in tqdm(range(num_matrices), desc="Generating frames"):

        # 1) Start with a completely zero image (requested)
        img = np.zeros((H, W), dtype=np.uint8)

        # Track which pixels are already used (avoid overlaps / reuse)
        occupied = np.zeros((H, W), dtype=bool)

        pairs_created = 0

        # 2) Sample pairs until we reach num_pairs
        #    We draw base pixels from Gaussian envelope and randomly pick orientation.
        #    We reject if:
        #      - neighbor goes out of bounds
        #      - either pixel is already used
        while pairs_created < num_pairs:

            flat_idx = np.random.choice(H * W, p=weights)
            y0 = flat_idx // W
            x0 = flat_idx % W

            o = np.random.randint(0, 2)
            dy, dx = orientations[o]
            y1 = y0 + dy
            x1 = x0 + dx

            # Neighbor must be inside bounds
            if not (0 <= y1 < H and 0 <= x1 < W):
                continue

            # No overlaps / no pixel reuse
            if occupied[y0, x0] or occupied[y1, x1]:
                continue

            # 3) Assign equal intensity to the pair
            if pair_value_mode == "fixed":
                v = fixed_pair_value
            else:
                v = np.random.randint(0, 256)

            img[y0, x0] = v
            img[y1, x1] = v

            occupied[y0, x0] = True
            occupied[y1, x1] = True
            pairs_created += 1

        # 4) Save this frame
        tif.write(img)

        # Save first frame for visualization
        if i == 0:
            first_img = img.copy()

# Show the first generated matrix
plt.figure(figsize=(5, 5))
plt.imshow(first_img, cmap="gray", vmin=0, vmax=255)
plt.colorbar()
plt.title("First frame (only correlated pixels non-zero)")
plt.tight_layout()
plt.show()
