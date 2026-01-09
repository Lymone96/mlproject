from dataclasses import dataclass

@dataclass
class Config:
    # ------------------- Paths ------------------------
    dataset_directory: str = "./data_storage/"
    checkpoints_directory: str = "./checkpoints/"
    model_filename: str = "model.pth"
    latent_filename: str = "latent.pth"
    train_new_model: bool = True
    save_network: bool = True 

    # -------------------- Dataset settings --------------------
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1

    # -------------------- Model settings --------------------
    latent_dim: int = 2
    input_values: int = 3
    number_of_hidden_layers: int = 4
    hidden_layers_neurons: int = 128
    output_values: int = 1
    activation_function: str = "relu"

    # -------------------- Training settings --------------------

    # Device
    device: str = "cuda"  # "cuda" or "cpu"

    # Optimization
    lr_model: float = 1e-3
    lr_latent: float = 1e-4

    # Loss
    latent_l2: float = 1e-4
    sdf_clamp: float = 5
    surface_w: float = 5.0
    surface_tau: float = 0.1

    # Sampling
    epochs: int = 50
    batch_size: int = 2048
    train_steps_per_epoch: int = 200
    validation_steps_per_epoch: int = 200

    # -------------------- Visualization settings --------------------
    per_axis_domain_length = 1 #From -0.5 to +0.5 by dataset settings
    per_axis_sample_number = 25 #Grid sample size (#x,#y,#z) is (100,100,100) by dataset settings
    add_reference_toggle = False
    reference_sample_name = "Cylinder-G25-0.txt"

