#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate N random 8-bit HxW images.

In each image:
- `num_groups` groups of correlated pixels.
- Each group has `k_corr` pixels (e.g., 10).
- Group anchor locations are drawn from a Gaussian envelope centered in the image.
- The other pixels in the group are at random distances (random dx, dy) from the anchor.
- All pixels within a group share identical intensity.
- Only correlated pixels are non-zero; all other pixels are exactly 0.
- No pixel is reused across groups (no overlaps).

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
H, W = 50, 50

k_corr = 10              # number of correlated pixels per group (was 2 before)
num_groups = 50          # number of such groups per image (YOU can choose this)

sigma = 8.0              # Gaussian envelope width (pixels)

# Random-distance correlation control
max_radius = 10          # max distance (in pixels) from anchor for group members (tunable)
min_radius = 1           # avoid placing everything on anchor; keep >=1

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

# Center of Gaussian
cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

# Gaussian weights over pixels
yy, xx = np.mgrid[0:H, 0:W]
gauss = np.exp(-(((yy - cy) ** 2) + ((xx - cx) ** 2)) / (2.0 * sigma ** 2))
weights = gauss.ravel()
weights /= weights.sum()

rng = np.random.default_rng()

with tiff.TiffWriter(out_path, bigtiff=True) as tif:
    first_img = None

    for i in tqdm(range(num_matrices), desc="Generating frames"):

        # 1) zero background (requested)
        img = np.zeros((H, W), dtype=np.uint8)
        occupied = np.zeros((H, W), dtype=bool)

        groups_done = 0

        # 2) Build groups
        while groups_done < num_groups:

            # Sample anchor pixel from Gaussian envelope
            flat_idx = rng.choice(H * W, p=weights)
            y0 = int(flat_idx // W)
            x0 = int(flat_idx % W)

            if occupied[y0, x0]:
                continue

            # Choose intensity for this group
            if pair_value_mode == "fixed":
                v = fixed_value
            else:
                v = int(rng.integers(0, 256))

            # Try to build a full group of k_corr distinct, in-bounds, non-overlapping pixels
            coords = [(y0, x0)]
            used_local = {(y0, x0)}  # avoid duplicates within group

            attempts = 0
            max_attempts = 5000  # prevents infinite loop if constraints are too tight

            while len(coords) < k_corr and attempts < max_attempts:
                attempts += 1

                # Random distance and random direction (dx,dy)
                r = int(rng.integers(min_radius, max_radius + 1))
                dy = int(rng.integers(-r, r + 1))
                dx = int(rng.integers(-r, r + 1))

                # reject zero move
                if dy == 0 and dx == 0:
                    continue

                y = y0 + dy
                x = x0 + dx

                # bounds check
                if not (0 <= y < H and 0 <= x < W):
                    continue

                # no overlap with already occupied pixels
                if occupied[y, x]:
                    continue

                # no duplicates inside same group
                if (y, x) in used_local:
                    continue

                coords.append((y, x))
                used_local.add((y, x))

            # If we failed to place enough members, reject this anchor and try again
            if len(coords) < k_corr:
                continue

            # Commit group pixels: mark occupied and write intensities
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
plt.title(f"First frame: {num_groups} groups × {k_corr} correlated pixels")
plt.tight_layout()
plt.show()
