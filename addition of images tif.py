import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tifffile as tiff

# Parameters
num_frames = 10000  # choose number of frames to be analyzed

Add = np.zeros((50, 50), dtype=np.float64)
number_of_files = 1

for ii in range(number_of_files):
    file_path = f'NF_GaussianEnvelope_SPDC_{ii}.tif'  # Specify your TIFF file path here

    with tiff.TiffFile(file_path) as tif:
        total_frames = min(len(tif.pages), num_frames)

        for i in tqdm(range(total_frames)):  # Loop over frames, ensuring enough frames for `frame2`
            frame1 = tif.pages[i].asarray().astype(np.float64)

            Add += frame1


fig, axs = plt.subplots(1, 2, figsize=(18, 5))

# --- Plot 1: RJDP ---
im0 = axs[0].imshow(Add, origin='lower')
axs[0].set_title('Nearfield y-axis JDP')
fig.colorbar(im0, ax=axs[0])

# --- Plot 3: Single frame ---
im2 = axs[1].imshow(frame1, origin='lower')
axs[1].set_title('Single Frame S=3')
fig.colorbar(im2, ax=axs[1])

plt.tight_layout()
plt.show()