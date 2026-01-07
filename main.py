import torch
from torch import nn
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Tuple

from model import DeepSDFModel
from config import Config
from dataset import load_txt_shapes

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

config = Config()

device = config.device

model = DeepSDFModel(
    config.input_values + config.latent_dim,
    config.number_of_hidden_layers,
    config.hidden_layers_neurons,
    config.output_values,
    config.activation_function,
)

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
    

dataset = ShapeDataset(Path(config.dataset_directory), device)
training_set, validation_set, test_set = torch.utils.data.random_split(dataset, [config.train_ratio, config.val_ratio, config.test_ratio])

print(len(training_set))
print(len(validation_set))
print(len(test_set))

latents = nn.Embedding(dataset.num_shapes, config.latent_dim)
nn.init.normal_(latents.weight, mean=0.0, std=0.01)

if not config.train_new_model:
    model.load_state_dict(torch.load(Path(config.checkpoints_directory, config.model_filename)))
    latents.load_state_dict(torch.load(Path(config.checkpoints_directory, config.latent_filename)))

model.to(device)
latents.to(device)

opt = torch.optim.Adam(
    [
        { "params": model.parameters(), "lr": config.lr_model },
        { "params": latents.parameters(), "lr": config.lr_latent },
    ]
)

writer = SummaryWriter("tensorboard")


def sdf_loss(pred, sdf, delta, w, tau):
    if delta is not None:
        pred = pred.clamp(-delta, delta)
        sdf = sdf.clamp(-delta, delta)
    a = sdf.abs()
    ww = 1.0 + w * torch.exp(-a / tau)
    return (ww * (pred - sdf).abs()).mean()

def latent_loss(latent, alpha):
    return alpha * latent.pow(2).mean()
    

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
        loss = sdf_loss(pred, t.unsqueeze(1), config.sdf_clamp, config.surface_w, config.surface_tau) + latent_loss(embedding, config.latent_l2)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        # opt.zero_grad()

        training_losses.append(loss.cpu().item())

    # Validation
    model.eval()
    validation_dataloader = DataLoader(validation_set, config.batch_size, shuffle=False)
    for i, ((x, s), t) in enumerate(validation_dataloader):
        if i == config.validation_steps_per_epoch:
            break

        embedding = latents(s)
        xyz_lambda = torch.hstack((x, embedding[:, 0, :]))
        pred = model(xyz_lambda)
        loss = sdf_loss(pred, t.unsqueeze(1), config.sdf_clamp, config.surface_w, config.surface_tau) + latent_loss(embedding, config.latent_l2)

        validation_losses.append(loss.mean().cpu().item())

    training_loss = np.array(training_losses).mean()
    validation_loss = np.array(validation_losses).mean()

    writer.add_scalar("Loss/training", training_loss, epoch)
    writer.add_scalar("Loss/validation", validation_loss, epoch)

    print(f"Epoch {epoch + 1}, Training: {training_loss:12.6f} | Validation: {validation_loss:12.6f}")

writer.close()