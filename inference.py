import numpy as np

from aitviewer.configuration import CONFIG as C
C.update_conf({"z_up": True})
# C.update_conf({"window_type": "pyglet"})

from aitviewer.renderables.volume import Volume
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.viewer import Viewer
import imgui
import sys
import pandas as pd
import re

from pathlib import Path
import torch
from torch import nn

from config import Config


if __name__ == "__main__":
    config = Config()
    device = config.device

    model = torch.load(Path(config.checkpoints_directory, config.model_filename), weights_only=False, map_location=torch.device(device))
    model.eval()

    latent_matrix_state_dict = torch.load(Path(config.checkpoints_directory, config.latent_filename), map_location=torch.device(device))
    num_of_shapes, lambda_dimension = latent_matrix_state_dict["weight"].shape

    # Add min and max for each lambda parameter
    ### Case 1: multiple training shapes, single lambda  
    # min_lambda = latent_matrix_state_dict["weight"].max().item()
    # max_lambda = latent_matrix_state_dict["weight"].min().item()

    ### Case 2: multiple training shapes, multidimensional lambda
    mins_lambda = [latent_matrix_state_dict["weight"][:, j].min().item() for j in range(lambda_dimension)]
    maxs_lambda = [latent_matrix_state_dict["weight"][:, j].max().item() for j in range(lambda_dimension)]

    latent_matrix = nn.Embedding(num_of_shapes, lambda_dimension).to(device)
    latent_matrix.load_state_dict(latent_matrix_state_dict)


    # Load reference training shape
    shape = (config.per_axis_sample_number, config.per_axis_sample_number, config.per_axis_sample_number)
    size = (config.per_axis_domain_length, config.per_axis_domain_length, config.per_axis_domain_length)
    level = 0.0

    # Data format: x = (100**3)
    x, y, z = np.meshgrid(
        np.linspace(-size[0] * 0.5, size[0] * 0.5, shape[0], dtype=np.float32), 
        np.linspace(-size[1] * 0.5, size[1] * 0.5, shape[1], dtype=np.float32),
        np.linspace(-size[2] * 0.5, size[2] * 0.5, shape[2], dtype=np.float32),
    )

    # Data format: pts = (100**3, 3)
    pts_np = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    pts = torch.from_numpy(pts_np).to(device)

    def gui_lambda_expanded():
        global sliders_values
        imgui.set_next_window_position(500, 50, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(250, 400, imgui.FIRST_USE_EVER)

        if imgui.begin("Lambda Control")[0]:
            updated_any = False

            # Sliders
            for i in range(lambda_dimension):
                slider_label = f"Lambda {i}"
                updated, new_value = imgui.slider_float(
                    slider_label,
                    sliders_values[i],
                    min_value=mins_lambda[i], # - 2,
                    max_value=maxs_lambda[i], # + 2,
                )
                if updated:
                    sliders_values[i] = new_value
                    updated_any = True

            imgui.separator()
            imgui.text("Presets (Trained Shapes)")

            # Buttons
            for k in range(num_of_shapes):
                button_label = f"Trained Shape {k}"

                if imgui.button(button_label):
                    # Retrieve the latent vector for index 'k'
                    with torch.no_grad():
                        # Use .tolist() to update the slider state
                        latent_vector = latent_matrix.weight[k].cpu().numpy()

                        # Update the global slider values list
                        sliders_values = latent_vector.tolist()
                        updated_any = True

            if updated_any:
                # Re-evaluate shape with new lambda
                with torch.no_grad():
                    # Prepare latents for the whole grid (pts.shape[0], lambda_dimension)
                    # Convert list back to tensor and broadcast to all points
                    current_latents_tensor = torch.tensor(sliders_values, device=device, dtype=torch.float32)
                    latents_grid = current_latents_tensor.repeat(pts.shape[0], 1)

                    samples = torch.hstack((pts, latents_grid))
                    new_pred = model(samples)

                    # Update volume in renderable
                    pred_vol.volume = new_pred.reshape(shape).cpu().numpy()
        imgui.end()

    slider_preset = 0.0
    sliders_values = [slider_preset for i in range(lambda_dimension)]

    # Reference selection and data loading
    if config.add_reference_toggle:

        sid = int(re.search(r"(\d+)(?!.*\d)", config.reference_sample_name).group(1))
        reference = np.array(pd.read_csv(Path("data_storage", config.reference_sample_name), skiprows=6).values[:, 3], np.float32).reshape(shape)

        # Extract latents from Embedding to match prediction to reference shape 
        latents = torch.empty(pts.shape[0], lambda_dimension).to(device)
        latents = latent_matrix(torch.full((pts.shape[0], 1), sid, dtype=torch.int32).to(device))[:, 0] #These indexing removes one extra dimension, which is artificially created 
        samples = torch.hstack((pts, latents)).to(device)

        with torch.no_grad():
            pred = model(samples)

        pred = pred.reshape(shape)

        # Here there are a few hardcoded parameters: x,y,z order and assumption that the domain is symmetrically distributed around the origin  
        ref_vol = Volume(reference, size, level, color=(0.5, 0, 0, 1.0), name="ref", position=(-size[0] * 0.5, -size[1] * 0.5, -size[2] * 0.5), max_triangles=int(10**6), max_vertices=int(10**6))

    else:
        latents = torch.empty(pts.shape[0], lambda_dimension).to(device)
        # Evaluate shape with preset slider values 
        with torch.no_grad():
            for j in range(lambda_dimension):
                latents[:, j] = torch.full((pts.shape[0],), sliders_values[j], dtype=torch.float32).to(device)
            samples = torch.hstack((pts, latents))
            pred = model(samples)

        pred = pred.reshape(shape)

    # Here there are a few hardcoded parameters: x,y,z order and assumption that the domain is symmetrically distributed around the origin  
    pred_vol = Volume(pred.cpu().numpy(), size, level, color=(0.0, 0.5, 0, 1.0), name="pred", position=(-size[0] * 0.5, -size[1] * 0.5, -size[2] * 0.5), max_triangles=int(10**6), max_vertices=int(10**6)) # position=(5, 0, 0))
    # print(pred.shape)

    # Visualize grid - Sanity check 
    # points = PointClouds(pts_np.reshape(1, -1, 3))

    v = Viewer()

    v.gui_controls["lambda vector"] = gui_lambda_expanded
    v.gui_controls.pop("playback", None)
    if config.add_reference_toggle:
        v.scene.add(ref_vol)
    v.scene.add(pred_vol)
    v.scene.add(points)
    v.scene.camera.position = (-2, 4, 10)
    v.scene.camera.target = (4, 1, 1)
    v.auto_set_camera_target = False
    v.run()