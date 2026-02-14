# -*- coding: utf-8 -*-
"""
Read input TIFF stack(s), apply a centered mask (3 short vertical lines -> set to 0),
and save the masked stack as a new .tif file.

Nothing else (no correlations, no plots).
"""

import numpy as np
from tqdm import tqdm
import tifffile as tiff

# -----------------------------
# Parameters
# -----------------------------
num_frames = 500000
H, W = 50, 50
number_of_files = 1

# Mask parameters
line_length = 12
line_thickness = 2
x_spacing = 4


# -----------------------------
# Build mask: 3 short vertical lines at center (ONCE)
# True  -> pixels to be zeroed
# False -> keep as-is
# -----------------------------
def make_three_vertical_lines_mask(H, W, line_length=12, line_thickness=2, x_spacing=4):
    mask = np.zeros((H, W), dtype=bool)

    cy = H // 2
    cx = W // 2

    y0 = max(0, cy - line_length // 2)
    y1 = min(H, cy + (line_length + 1) // 2)

    x_centers = [cx - x_spacing, cx, cx + x_spacing]
    half_t = line_thickness // 2

    for xc in x_centers:
        x0 = max(0, xc - half_t)
        x1 = min(W, xc + half_t + 1)
        mask[y0:y1, x0:x1] = True

    return mask


mask3 = make_three_vertical_lines_mask(H, W, line_length=line_length,
                                       line_thickness=line_thickness, x_spacing=x_spacing)

# -----------------------------
# Main: read -> mask -> write
# -----------------------------
for ii in range(number_of_files):
    in_path = f'NF_GaussianEnvelope_SPDC_{ii}.tif'
    out_path = f'Masked_NF_GaussianEnvelope_SPDC_{ii}.tif'

    with tiff.TiffFile(in_path) as tif:
        total_frames = min(len(tif.pages), num_frames)

        # Use dtype of input to preserve format (e.g., uint8/uint16/float32)
        sample = tif.pages[0].asarray()
        out_dtype = sample.dtype

        # Create writer for output stack
        with tiff.TiffWriter(out_path, bigtiff=True) as tw:
            for i in tqdm(range(total_frames), desc=f"Masking {in_path}"):
                frame = tif.pages[i].asarray()

                # Ensure expected shape (optional safety)
                if frame.shape != (H, W):
                    raise ValueError(f"Frame shape {frame.shape} != expected {(H, W)}")

                # Work in float for masking, then cast back
                frame_f = frame.astype(np.float64, copy=False)
                frame_f[mask3] = 0.0

                tw.write(frame_f.astype(out_dtype, copy=False))

    print(f"Saved masked stack: {out_path}")
