#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 1e6 random 8-bit 64x64 images.
In each image: 100 nearest-neighbor pixel pairs (horizontal or vertical) have equal intensity.
Pair locations are drawn randomly from a Gaussian envelope centered in the image.
Each image has a different random set of pairs.

@author: satyajeet
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

# -----------------------------
# Parameters
# -----------------------------
num_matrices = 100000
H, W = 20, 20

num_pairs = 100                 # total paired pixels per image
sigma = 8.0                     # Gaussian envelope width (pixels). Adjust as needed.
pair_value_mode = "random"      # "random" or "fixed"
fixed_pair_value = 250          # used only if pair_value_mode == "fixed"

# Background random image (like detector noise / randomness)
background_mode = "uniform"     # "uniform" or "zeros"
# -----------------------------

# Center of Gaussian (use center of image grid)
cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

# Precompute Gaussian weights over the whole image
yy, xx = np.mgrid[0:H, 0:W]
gauss = np.exp(-(((yy - cy) ** 2) + ((xx - cx) ** 2)) / (2.0 * sigma ** 2))
weights = gauss.ravel()
weights /= weights.sum()  # probability distribution over pixels

# Neighbor offsets: right (0,+1) and down (+1,0)
# We'll randomly choose orientation for each pair.
# (dy, dx) arrays will be built per image.
orientations = np.array([[0, 5], [5, 0]], dtype=np.int32)  # horizontal, vertical

# Write frames one-by-one to avoid massive RAM
with tiff.TiffWriter("NF_GaussianEnvelope_SPDC_0.tif", bigtiff=True) as tif:

    for i in range(num_matrices):

        # 1) Start with background
        if background_mode == "uniform":
            img = np.random.randint(0, 256, size=(H, W), dtype=np.uint8)
        else:
            img = np.zeros((H, W), dtype=np.uint8)

        # 2) Sample candidate base pixels from Gaussian envelope
        #    We oversample then filter out invalid neighbor pairs.
        #    This avoids edge issues where neighbor would go out of bounds.
        #    Oversample factor ~2–4 is usually enough.
        need = num_pairs
        bases_y = []
        bases_x = []
        dy_list = []
        dx_list = []

        while need > 0:
            draw = max(need * 4, 200)  # oversample chunk
            flat_idx = np.random.choice(H * W, size=draw, replace=True, p=weights)
            y0 = flat_idx // W
            x0 = flat_idx % W

            # Random orientation for each candidate base
            o = np.random.randint(0, 2, size=draw)
            dy = orientations[o, 0]
            dx = orientations[o, 1]

            y1 = y0 + dy
            x1 = x0 + dx

            # Keep only those where neighbor is inside image
            ok = (y1 >= 0) & (y1 < H) & (x1 >= 0) & (x1 < W)

            y0_ok = y0[ok]
            x0_ok = x0[ok]
            dy_ok = dy[ok]
            dx_ok = dx[ok]

            take = min(need, y0_ok.size)
            if take > 0:
                bases_y.append(y0_ok[:take])
                bases_x.append(x0_ok[:take])
                dy_list.append(dy_ok[:take])
                dx_list.append(dx_ok[:take])
                need -= take

        y0 = np.concatenate(bases_y)
        x0 = np.concatenate(bases_x)
        dy = np.concatenate(dy_list)
        dx = np.concatenate(dx_list)

        y1 = y0 + dy
        x1 = x0 + dx

        # 3) Assign equal intensities to the two pixels in each pair
        if pair_value_mode == "fixed":
            v = np.full(num_pairs, fixed_pair_value, dtype=np.uint8)
        else:
            v = np.random.randint(0, 256, size=num_pairs, dtype=np.uint8)

        img[y0, x0] = v
        img[y1, x1] = v

        # 4) Save this frame
        tif.write(img)

        # Optional: visualize only the first frame
        if i == 0:
            first_img = img.copy()

# Show the first generated matrix
plt.imshow(first_img, cmap="gray")
plt.colorbar()
plt.title("First frame (Gaussian-distributed nearest-neighbor equal-intensity pairs)")
plt.show()
