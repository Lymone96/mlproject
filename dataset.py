from pathlib import Path

import os
import re
import pandas as pd
import torch

def load_txt_shapes(folder: Path):
    shapes = {}
    for f in folder.iterdir():
        if not f.name.lower().endswith(".txt"):
            continue

        key = os.path.splitext(f.name)[0]
        number_match = re.search(r"(\d+)(?!.*\d)", key).group(1)
        sid = int(number_match)
        data = torch.tensor(pd.read_csv(f, skiprows=6).values, dtype=torch.float32)
        shapes[key] = (data, sid)  # {"ShapeName-G##-IDNumber": (DataTensor, IDNumber)}
    return shapes
