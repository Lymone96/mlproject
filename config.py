from dataclasses import dataclass

@dataclass
class Config:
    # ------------------- Paths ------------------------
    dataset_directory: str = r"C:\Users\gijs\OneDrive - Delft University of Technology\Bestanden van Gabriele Mylonopoulos - CIEM2003 - WIND Group\003 - Training Datasets and 3D Models - by stage\Exp 1b - One design, multiple parameters variation\E1b - Dataset"
    checkpoints_directory: str = r"C:\Users\gijs\tu_delft_local\2025-2026\Q2\DSAIE\models"
    model_filename: str = "model.pth"
    latent_filename: str = "latent.pth"
    train_new_model: bool = True
    save_network: bool = True

    # -------------------- Dataset settings --------------------
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1

    # -------------------- Model settings --------------------
    latent_dim: int = 1
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
    surface_w: float = 5


    # Sampling
    epochs: int = 100
    batch_size: int = 2048
    train_steps_per_epoch: int = 200
    validation_steps_per_epoch: int = 200

    # -------------------- Visualization settings --------------------
    per_axis_domain_length: int = 1  #From -0.5 to +0.5 by dataset settings
    per_axis_sample_number: int = 100 #Grid sample size (#x,#y,#z) is (100,100,100) by dataset settings
    add_reference_toggle: bool = True
    reference_sample_name: str = "WindFloat-1.txt"

