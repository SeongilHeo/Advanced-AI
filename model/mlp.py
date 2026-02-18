import torch.nn as nn
from typing import Callable, Dict, Optional, Sequence, Union

ActivationFactory = Callable[[], nn.Module]

activation_table: Dict[str, ActivationFactory] = {
    "identity": nn.Identity,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
}

class MLP(nn.Module):
    """Simple configurable MLP.

    Builds an MLP of the form:
        Linear -> act -> (Linear -> act)* -> Linear -> out_act(optional)

    Notes:
        - `hidden_dims` must contain at least one hidden layer size.
        - `output_activation=None` disables the output activation.
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int],
            output_dim: int,
            activation: str = "leaky_relu",
            output_activation: Optional[Union[str, None]] = "identity"
        ) -> None:
        super().__init__()

        if activation not in activation_table:
            raise ValueError(f"Unknown activation: {activation}")
        if output_activation and output_activation not in activation_table:
            raise ValueError(f"Unknown output_activation: {output_activation}")
        
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")
        
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims)-1)]
        )
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.activation = activation_table[activation]()
        self.output_activation = activation_table[output_activation]() if output_activation else None

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        out = self.output_layer(x)
        if self.output_activation:
            out = self.output_activation(out)

        return out
    
