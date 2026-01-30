from torch import nn
from pathlib import Path
from typing import Dict, Tuple
from model import DeepSDFModel
from config import Config
from dataset import load_txt_shapes
from torch.utils.data import Dataset, DataLoader

import copy
import torch
import numpy as np

config = Config()

# Early stopping
best_validation_loss = float("inf")
passed_epochs = 0
best_model = ""
best_latents = ""

if config.cuda_manual_seed == True:
    torch.cuda.manual_seed_all(seed=42)

device = config.device

# Define classes and functions
class ShapeDataset(Dataset):
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


def get_sdf_loss(pred_sdf, target_sdf, w):
    a = target_sdf.abs()
    ww = 1.0 + w * torch.exp(-a)
    return (ww * (pred_sdf - target_sdf).abs()).mean()


def get_latent_loss(latent, L2_regularization):
    return L2_regularization * latent.pow(2).mean()

# Get model architecture
model = DeepSDFModel(
    config.input_values + config.latent_dim,
    config.number_of_hidden_layers,
    config.hidden_layers_neurons,
    config.output_values,
    config.hidden_activation_function,
    config.output_activation_function,
)

# Importing dataset
dataset = ShapeDataset(Path(config.dataset_directory), device)

# Splitting data set
training_set, validation_set, test_set = torch.utils.data.random_split(dataset, [config.train_ratio, config.val_ratio, config.test_ratio])

# Initialize latent space
latents = nn.Embedding(dataset.num_shapes, config.latent_dim)
nn.init.normal_(latents.weight, mean=0.0, std=1)

# Import old model if desired
if not config.train_new_model:
    model.load_state_dict(torch.load(Path(config.checkpoints_directory_model, config.model_filename)))
    latents.load_state_dict(torch.load(Path(config.checkpoints_directory_latent, config.latent_filename)))

# Set model to device
model.to(device)
latents.to(device)

# Initialize optimizer
opt = torch.optim.Adam(
    [
        { "params": model.parameters(), "lr": config.lr_model },
        { "params": latents.parameters(), "lr": config.lr_latent },
    ]
)

# Start training loop
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
        total_training_loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        
        
        training_losses.append(total_training_loss.cpu().item())

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

        validation_losses.append(total_validation_loss.mean().cpu().item())

    training_loss = np.array(training_losses).mean()
    validation_loss = np.array(validation_losses).mean()

    print(f"Epoch {epoch + 1}, Training: {training_loss:12.6f} | Validation: {validation_loss:12.6f}")

    # Early stopping
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

# Model Export
def save_network_and_latents(model, latents):

    torch.save(model, Path(config.checkpoints_directory_model, config.model_filename))
    torch.save(latents.state_dict(), Path(config.checkpoints_directory_latent, config.latent_filename))

if config.save_network:
    save_network_and_latents(
        model=model,
        latents=latents
    )