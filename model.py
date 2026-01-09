from torch import nn

def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "gelu":
        return nn.GELU()
    elif name == "leakyrelu":
        return nn.LeakyReLU()
    elif name == "silu" or name == "swish":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation: {name}")


class DeepSDFModel(nn.Module):
    """
    Flexible nonlinear MLP:

    Inputs:
        input_values (int)              - size of input vector
        number_of_hidden_layers (int)   - how many hidden layers
        hidden_layers_neurons (int)     - neurons per hidden layer
        output_values (int)             - size of output vector
        activation_function (str)       - e.g. 'relu', 'tanh', 'gelu'
    """

    def __init__(
            self,
            input_values: int,
            number_of_hidden_layers: int,
            hidden_layers_neurons: int,
            output_values: int,
            activation_function: str,
    ):
        super().__init__()

        activation = get_activation(activation_function)

        layers = []

        # Input → first hidden layer
        layers.append(nn.Linear(input_values, hidden_layers_neurons))
        layers.append(activation)

        # Hidden layers
        for _ in range(number_of_hidden_layers - 1):
            layers.append(nn.Linear(hidden_layers_neurons, hidden_layers_neurons))
            layers.append(activation)

        # Last layer → output
        layers.append(nn.Linear(hidden_layers_neurons, output_values))

        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz_lambda):
        return self.mlp(xyz_lambda)