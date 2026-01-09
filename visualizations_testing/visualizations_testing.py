

# THIS FILE IS A COPY FROM THE PREVIOUS IMPLEMENTATION - Repo: DeepSDF-merging_santino_gijs

import numpy as np

from aitviewer.configuration import CONFIG as C
C.update_conf({"z_up": True})

from aitviewer.renderables.volume import Volume
from aitviewer.viewer import Viewer
import sys
import pandas as pd


if __name__ == "__main__":
    # Load a signed distance field from a file and mesh with a marching cubes algorithm implemented in a GPU shader.
    # volume: np.ndarray = np.load("/home/cashlemon/Downloads/dragon.npz")["volume"]
    # SIZE = np.array(volume.shape[::-1], np.float32) / max(volume.shape) * 6
    # LEVEL = 0.0

    # Hardcoded parameters
    shape = (100, 100, 100)
    size = (1, 1, 1)
    level = 0.0

    data = np.array(pd.read_csv(sys.argv[1], skiprows=6).values[:, 3], np.float32).reshape(shape).swapaxes(0, 2)
    vol = Volume(data, size, level)
    # vol = Volume(volume, SIZE, LEVEL)


    v = Viewer()
    v.scene.add(vol)
    v.scene.camera.position = (-2, 4, 10)
    v.scene.camera.target = (4, 1, 1)
    v.auto_set_camera_target = False
    v.run()