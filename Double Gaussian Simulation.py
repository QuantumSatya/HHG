import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.ndimage import gaussian_filter
from tifffile import TiffWriter

# ---------------------------
# Grid setup
# ---------------------------
N = 20
x = np.linspace(-5, 5, N)
y = np.linspace(-5, 5, N)
X, Y = np.meshgrid(x, y)

num_frames = int(1e6)
plotter = False  # WARNING: do not plot 1e6 frames

# ---------------------------
# Base Gaussian parameters
# ---------------------------
A = 1.0
sigma_x, sigma_y = 1.0, 1.2

# Separation factor (dimensionless)
# Definition: d = S * sigma_x
S = 3
d = S * sigma_x

# Two Gaussian centers (symmetric about origin)
x01, y01 = -d / 2.0, 0.0
x02, y02 = +d / 2.0, 0.0

# Optional: random jitter of the centers per frame (set 0 to disable)
pixelShift = 0.03  # in same units as x,y grid (NOT pixels)
use_jitter = False

# ---------------------------
# Noise / distortion parameters
# ---------------------------
# Method 1: multiplicative spatially-correlated noise
noise_strength = 0.5       # 0.5 -> 50% variation
noise_smooth_sigma = 2     # gaussian_filter sigma

# Method 2: pixelwise sigma variation field
sigma_var_strength = 0.1   # 10% pixelwise sigma variations

# Mixing between the two methods:
# alpha=1 => only Method 1 result
# alpha=0 => only Method 2 result
alpha = 0.0  # set to 0.5 if you want a true mix

# ---------------------------
# Output TIFF (streamed: no giant RAM cube)
# ---------------------------
tif_filename = "gaussian_50percent_method_2_only_G_local_cube_0.tif"

# Optional: quick theory overlap (1D equal-sigma reference)
overlap_theory = np.exp(-(S**2) / 4.0)
print(f"S = {S}, d = {d}")
print(f"Theoretical 1D overlap exp(-S^2/4) = {overlap_theory:.6f}")

# If you want a small preview grid of a few frames (recommended)
preview_frames = 10 if plotter else 0
preview = []

with TiffWriter(tif_filename, bigtiff=True) as tif:
    for k in tqdm.tqdm(range(num_frames)):
        # Optional random jitter per frame
        if use_jitter:
            dx1, dy1 = np.random.uniform(-pixelShift, pixelShift, size=2)
            dx2, dy2 = np.random.uniform(-pixelShift, pixelShift, size=2)
            cx1, cy1 = x01 + dx1, y01 + dy1
            cx2, cy2 = x02 + dx2, y02 + dy2
        else:
            cx1, cy1 = x01, y01
            cx2, cy2 = x02, y02

        # --- Ideal: sum of two Gaussians ---
        G1 = A * np.exp(-((X - cx1) ** 2 / (2 * sigma_x ** 2) +
                          (Y - cy1) ** 2 / (2 * sigma_y ** 2)))
        G2 = A * np.exp(-((X - cx2) ** 2 / (2 * sigma_x ** 2) +
                          (Y - cy2) ** 2 / (2 * sigma_y ** 2)))
        G = G1 + G2

        # ---- Method 1: multiplicative, spatially-correlated noise ----
        noise = np.random.normal(0, 1, size=G.shape)
        noise_smooth = gaussian_filter(noise, sigma=noise_smooth_sigma)
        G_m1 = G * (1 + noise_strength * noise_smooth)
        G_m1 = np.clip(G_m1, 0, None)  # keep non-negative

        # ---- Method 2: pixelwise varying sigma (distorted Gaussian) ----
        sigma_variation = sigma_var_strength * np.random.normal(size=G.shape)
        sigma_x_local = np.clip(sigma_x * (1 + sigma_variation), 1e-6, None)
        sigma_y_local = np.clip(sigma_y * (1 + sigma_variation), 1e-6, None)

        G1_local = np.exp(-((X - cx1) ** 2 / (2 * sigma_x_local ** 2) +
                            (Y - cy1) ** 2 / (2 * sigma_y_local ** 2)))
        G2_local = np.exp(-((X - cx2) ** 2 / (2 * sigma_x_local ** 2) +
                            (Y - cy2) ** 2 / (2 * sigma_y_local ** 2)))
        G_m2 = G1_local + G2_local

        # ---- Mix both methods (fixed; no accidental "*0") ----
        off_gaussian = alpha * G_m1 + (1 - alpha) * G_m2

        # Normalize to [0,1]
        m = np.max(off_gaussian)
        if m > 0:
            off_gaussian = off_gaussian / m
        off_gaussian = off_gaussian.astype(np.float32)

        # Stream-write each frame (no giant data_cube in RAM)
        tif.write(off_gaussian, photometric="minisblack")

        # Store a few frames for preview plotting
        if plotter and k < preview_frames:
            preview.append(off_gaussian)

print(f"✅ Saved {num_frames} frames to '{tif_filename}'")

# ---------------------------
# Optional preview plotting (only a few frames)
# ---------------------------
if plotter and len(preview) > 0:
    cols = 5
    rows = int(np.ceil(len(preview) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows))
    axes = np.array(axes).ravel()

    for i in range(len(preview)):
        axes[i].imshow(preview[i], origin="lower", cmap="viridis")
        axes[i].set_title(f"Frame {i}")
        axes[i].axis("off")

    for j in range(len(preview), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
