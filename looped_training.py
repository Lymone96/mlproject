from pathlib import Path
from typing import Tuple, Dict
from torch import nn
from torch.utils.data import DataLoader, Dataset
from config import Config
from dataset import load_txt_shapes
from model import DeepSDFModel

import copy
import numpy as np
import pandas as pd
import torch

config = Config()
patience = 5  # Number of epochs without improvement before stopping


class ShapeDataset_training_loop(Dataset):
    def __init__(self, directory: Path, device):
        shapes: Dict[str, Tuple[torch.Tensor, int]] = load_txt_shapes(directory)
        self.num_shapes = len(shapes)

        xyz_samples = []
        sid_samples = []
        targets = []
        for xyz_sdf, sid in shapes.values():
            sids = torch.full((xyz_sdf.shape[0], 1), sid)
            sid_samples.append(sids)
            xyz_samples.append(xyz_sdf[:, :3])
            targets.append(xyz_sdf[:, 3])

        self.xyz_samples = torch.concat(xyz_samples).to(device)
        self.sid_samples = torch.concat(sid_samples).to(device)
        self.targets = torch.concat(targets).to(device)

    def __len__(self):
        return self.xyz_samples.shape[0]

    def __getitem__(self, index):
        return (self.xyz_samples[index], self.sid_samples[index]), self.targets[index]


if __name__ == "__main__":
    torch.cuda.manual_seed_all(seed=42)

    rows = []
    # surface_w_values = [5, 7, 9, 11, 20, 50]
    number_of_neurons = [512, 256, 128]
    number_of_layers = [8, 6, 4]
    for number_of_layer in number_of_layers:
        for number_of_neuron in number_of_neurons:

            config.number_of_hidden_layers = number_of_layer
            config.hidden_layers_neurons = number_of_neuron

            passed_epochs = 0
            best_validation_loss = float('inf')  # Initialize to a very large value
            patience_counter = 0  # Counter to track the number of epochs without improvement

            print(f"start test for : {number_of_layer} layers and {number_of_neuron} neurons")
            # config.latent_dim = latent_dimension
            model = DeepSDFModel(
                config.input_values + config.latent_dim,
                config.number_of_hidden_layers,
                config.hidden_layers_neurons,
                config.output_values,
                config.activation_function,
            )
            device = config.device

            dataset = ShapeDataset_training_loop(Path(config.dataset_directory), device)
            training_set, validation_set, test_set = torch.utils.data.random_split(dataset,
                                                                                   [config.train_ratio, config.val_ratio,
                                                                                    config.test_ratio])



            latents = nn.Embedding(dataset.num_shapes, config.latent_dim)
            nn.init.normal_(latents.weight, mean=0.0, std=1)

            model.to(device)
            latents.to(device)

            opt = torch.optim.Adam(
                [
                    {"params": model.parameters(), "lr": config.lr_model},
                    {"params": latents.parameters(), "lr": config.lr_latent},
                ]
            )


            def get_sdf_loss(pred_sdf, target_sdf, w):
                a = target_sdf.abs()
                ww = 1.0 + w * torch.exp(-a)
                return (ww * (pred_sdf - target_sdf).abs()).mean()


            def get_latent_loss(latent, L2_regularization):
                return L2_regularization * latent.pow(2).mean()


            training_loss_history = []
            validation_loss_history = []

            for epoch in range(config.epochs):
                training_losses = []
                validation_losses = []

                opt.zero_grad()

                # Training
                model.train()
                training_dataloader = DataLoader(training_set, config.batch_size, shuffle=True)
                for i, ((x, s), t) in enumerate(training_dataloader):
                    if i == config.train_steps_per_epoch:
                        break

                    embedding = latents(s)
                    xyz_lambda = torch.hstack((x, embedding[:, 0, :]))
                    pred = model(xyz_lambda)

                    sdf_loss = get_sdf_loss(
                        pred_sdf=pred,
                        target_sdf=t.unsqueeze(1),
                        w=config.surface_w
                    )

                    latent_loss = get_latent_loss(
                        latent=embedding,
                        L2_regularization=config.latent_l2
                    )

                    total_training_loss = sdf_loss + latent_loss
                    opt.step()
                    opt.zero_grad(set_to_none=True)


                    training_losses.append(total_training_loss.detach().cpu().item())

                # Validation
                model.eval()
                validation_dataloader = DataLoader(validation_set, config.batch_size, shuffle=False)
                for i, ((x, s), t) in enumerate(validation_dataloader):
                    if i == config.validation_steps_per_epoch:
                        break

                    embedding = latents(s)
                    xyz_lambda = torch.hstack((x, embedding[:, 0, :]))
                    pred = model(xyz_lambda)
                    sdf_loss = get_sdf_loss(
                        pred_sdf=pred,
                        target_sdf=t.unsqueeze(1),
                        w=config.surface_w
                    )
                    latent_loss = get_latent_loss(
                        latent=embedding,
                        L2_regularization=config.latent_l2
                    )
                    total_validation_loss = sdf_loss + latent_loss


                validation_losses.append(total_validation_loss.detach().cpu().item())
                training_loss = float(np.mean(training_losses)) if training_losses else float("nan")
                validation_loss = float(np.mean(validation_losses)) if validation_losses else float("nan")

                rows.append({
                    "layers": config.number_of_hidden_layers,
                    "neurons": config.hidden_layers_neurons,
                    "epoch": epoch + 1,
                    "train_loss": float(training_loss),
                    "val_loss": float(validation_loss),
                    "batch_size": config.batch_size,
                    "lr_model": config.lr_model,
                    "lr_latent": config.lr_latent,
                    "latent_dim": config.latent_dim,
                })

                print(f"Epoch {epoch + 1}, Training: {training_loss:12.6f} | Validation: {validation_loss:12.6f}")
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_model = copy.deepcopy(model)
                    best_latents = copy.deepcopy(latents)
                    passed_epochs = 0
                else:
                    passed_epochs += 1
                    if passed_epochs >= config.max_passed_epochs:
                        print(f"early stopping after {epoch - passed_epochs} epochs with {best_validation_loss:12.6f}")
                        model = best_model
                        latents = best_latents
                        break


            directory = Path(config.looped_training_directory)

            folder_name = f"network_with_{number_of_layer}_layer_{number_of_neuron}_neuron"

            out_dir = directory / folder_name
            out_dir.mkdir(parents=True, exist_ok=True)

            model_filename = "model.pth"
            latent_filename = "latent.pth"

            model_path = out_dir / model_filename
            latent_path = out_dir / latent_filename

            torch.save(model, model_path)
            torch.save(latents.state_dict(), latent_path)

            file_name = f"for_epochs_latent_dim.pickle"
            file_path = directory / file_name

            df = pd.DataFrame(rows)
            df.to_pickle(file_path)
