# Dependencies
- This project is mainly based on `pytorch`. `numpy` and `pandas` are also used for data handling and manipulation.
- AIT viewer - https://github.com/eth-ait/aitviewer  
	It may be worth mentioning that the latest version produced a few issues that were solved by installing the dev branch through the following command:
	`pip install git+https://github.com/eth-ait/aitviewer.git@dev` 

# Content of this repository
- `config.py` collects the main properties that the user may want to customize when setting up training of a new model, loading and storage locations and visualization.
- `main.py` contains the implementation of training loop
- `model.py` contains the implementation of the model architecture, as defined in the configuration file
- `dataset.py`contains the data ingestion function
- `inference.py` includes the prediction and visualization functions used for reconstruction assessment of the selected model
- `looped_training.py` - discontinued - the purpose of this script was evaluating multiple architecture configurations  
    *the file is included to illustrate the progression of the project but the training loop in this file is not updated to the latest tuning*
- Folder structure:  
	- `checkpoints` - default location for model and latents saving
	- `data_storage` - default location to place dataset (`.txt` files only) for training
	- `data_backup` - utility folder to temporarily store or move data files
	- `looped_traininig_results` - default location to save results from `looped_training.py` 

# Dataset generation and experiments
- For the purposes of the project the models were trained in a set of experiments. The pertinent datasets and reference `.obj` models can be found at this link: 
- Naming convention:  
  `shape_name-grid_resolution-incremental_number.txt`  
  `shape_name-grid_resolution-incremental_number.obj`  
- Grasshopper scripts used in the generation of the datasets and 3D models are located in a dedicated folder at the previous link  
*Disclaimer: links are not permanently maintained, they're intended for assessment from course staff*

# (Pre)Trained Models
- Main models produced during experiments can be found at this link:  
*Disclaimer: links are not permanently maintained, they're intended for assessment from course staff*

# Usage of this repository
**Training**
- Set desired properties in `config.py`.
- Populate `data_storage` folder with desired `.txt` files
- Run `main.py` to initiate training  
**Model Saving**
- If the `save_network` is set to `True`, the script exports both model and latents' embedding according to the naming defined in `config.py`. Default settings are `model.pth` and `latents.pth` exported to the `./checkpoints/` folder.  
**Inference**
- Set visualization properties in the dedicated section of `config.py`. Special attention should be paid to matching the grid discretization of training files and visualization settings.  
	For the provided dataset:
    - `per_axis_domain_lenght` is always 1. 
    - `per_axis_sample_number` can be found in the file name according to the above mentioned convention.
- A reference file from training dataset can also be added through `reference_sample_name`
