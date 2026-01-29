import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ------------------- Config-like path -------------------
dataset_directory = "./003 - Training Datasets and 3D Models - by stage/Exp 1b - One design, multiple parameters variation/E1b - Dataset/"   # matches your config.py

# ------------------- Helper: read one txt file -------------------
def load_sdf_from_txt(txt_path: Path):
    """
    Reads your dataset txt file format:
    header lines...
    then a line: X,Y,Z,SDF
    then numeric rows: x,y,z,sdf
    """
    sdf_values = []
    with open(txt_path, "r") as f:
        lines = f.readlines()

    # find the line where actual data starts
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "X,Y,Z,SDF":
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError(f"File {txt_path} does not contain 'X,Y,Z,SDF' header.")

    # parse numeric rows
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 4:
            continue
        sdf_values.append(float(parts[3]))

    return np.array(sdf_values)


# ------------------- Load all SDF values from folder -------------------
dataset_directory = Path(dataset_directory)

all_sdf = []
txt_files = sorted(dataset_directory.glob("*.txt"))

print(f"Found {len(txt_files)} txt files in {dataset_directory}")

for fp in txt_files:
    sdf = load_sdf_from_txt(fp)
    all_sdf.append(sdf)
    print(f"{fp.name}: loaded {len(sdf)} sdf values, min={sdf.min():.4f}, max={sdf.max():.4f}")

all_sdf = np.concatenate(all_sdf)
print(f"\nTOTAL points: {len(all_sdf)}")
print(f"Global min SDF = {all_sdf.min():.4f}")
print(f"Global max SDF = {all_sdf.max():.4f}")

# ------------------- Histogram with custom bins -------------------
bin_width = 0.1
bins = np.arange(np.floor(all_sdf.min()), np.ceil(all_sdf.max()) + bin_width, bin_width)

counts, edges = np.histogram(all_sdf, bins=bins)

print("\nHistogram bin counts:")
for i in range(len(counts)):
    print(f"[{edges[i]:6.2f}, {edges[i+1]:6.2f}): {counts[i]}")

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(all_sdf, bins=bins)
plt.xlabel("SDF value")
plt.ylabel("Number of points")
plt.title("Histogram of SDF values (all shapes)")
plt.grid(True)
plt.show()
