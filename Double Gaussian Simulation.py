#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate N random 8-bit HxW images.

In each image:
- num_groups groups of correlated pixels
- each group has k_corr pixels
- group anchors sampled from TWO-GAUSSIAN envelope separated by s pixels
- remaining group pixels placed at random distances from anchor (dx,dy)
- all pixels in group share identical intensity
- background is exactly zero
- no overlaps across groups
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from tqdm import tqdm

# -----------------------------
# Parameters
# -----------------------------
num_matrices = 500000
H, W = 50, 50

k_corr = 15
num_groups = 5

sigma = 3.0              # Gaussian envelope width (pixels)

# Separation between the two Gaussian centers (in pixels)
s = 20                   # <<< CHANGE THIS (pixels)

# Random-distance correlation control
max_radius = 8
min_radius = 1

pair_value_mode = "random"  # "random" or "fixed"
fixed_value = 250

out_path = "NF_GaussianEnvelope_SPDC_0.tif"
# -----------------------------

# Sanity check: total correlated pixels per image cannot exceed H*W if no overlaps
total_corr_pixels = num_groups * k_corr
if total_corr_pixels > H * W:
    raise ValueError(
        f"num_groups*k_corr = {total_corr_pixels} exceeds total pixels {H*W} "
        f"for {H}x{W} without overlap."
    )

# Center of image
cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

# Two Gaussian centers separated by s pixels (along x direction)
cx1, cy1 = cx - s / 2.0, cy
cx2, cy2 = cx + s / 2.0, cy

# Gaussian weights over pixels (two lobes)
yy, xx = np.mgrid[0:H, 0:W]
gauss1 = np.exp(-(((yy - cy1) ** 2) + ((xx - cx1) ** 2)) / (2.0 * sigma ** 2))
gauss2 = np.exp(-(((yy - cy2) ** 2) + ((xx - cx2) ** 2)) / (2.0 * sigma ** 2))
gauss = gauss1 + gauss2

weights = gauss.ravel().astype(np.float64)
weights /= weights.sum()

rng = np.random.default_rng()

with tiff.TiffWriter(out_path, bigtiff=True) as tif:
    first_img = None

    for i in tqdm(range(num_matrices), desc="Generating frames"):

        # 1) zero background
        img = np.zeros((H, W), dtype=np.uint8)
        occupied = np.zeros((H, W), dtype=bool)

        groups_done = 0

        # 2) Build groups
        while groups_done < num_groups:

            # Sample anchor pixel from TWO-GAUSSIAN envelope
            flat_idx = rng.choice(H * W, p=weights)
            y0 = int(flat_idx // W)
            x0 = int(flat_idx % W)

            if occupied[y0, x0]:
                continue

            # Choose intensity for this group
            if pair_value_mode == "fixed":
                v = int(fixed_value)
            else:
                v = int(rng.integers(0, 256))

            # Try to build a full group of k_corr distinct, in-bounds, non-overlapping pixels
            coords = [(y0, x0)]
            used_local = {(y0, x0)}

            attempts = 0
            max_attempts = 5000

            while len(coords) < k_corr and attempts < max_attempts:
                attempts += 1

                rr = int(rng.integers(min_radius, max_radius + 1))
                dy = int(rng.integers(-rr, rr + 1))
                dx = int(rng.integers(-rr, rr + 1))

                if dy == 0 and dx == 0:
                    continue

                y = y0 + dy
                x = x0 + dx

                if not (0 <= y < H and 0 <= x < W):
                    continue
                if occupied[y, x]:
                    continue
                if (y, x) in used_local:
                    continue

                coords.append((y, x))
                used_local.add((y, x))

            if len(coords) < k_corr:
                continue

            # Commit group
            for (y, x) in coords:
                img[y, x] = v
                occupied[y, x] = True

            groups_done += 1

        # 3) Save frame
        tif.write(img)

        if i == 0:
            first_img = img.copy()

# Visualize first frame
plt.figure(figsize=(5, 5))
plt.imshow(first_img, cmap="gray", vmin=0, vmax=255)
plt.colorbar()
plt.title(f"First frame: {num_groups} groups × {k_corr} correlated pixels\n"
          f"Two Gaussians separated by s={s} px, sigma={sigma} px")
plt.tight_layout()
plt.show()
