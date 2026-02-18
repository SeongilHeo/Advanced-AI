import torch.nn as nn
from typing import Callable, Dict, Optional

ActivationFactory = Callable[[], nn.Module]

activation_table: Dict[str, ActivationFactory] = {
    "identity": nn.Identity,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
}

class FCNet(nn.Module):
    '''
        Neural network with that maps (s,s') state to a prediction
        over which of the three discrete actions was taken.
        The network should have three outputs corresponding to the logits for a 3-way classification problem.
    '''
    def __init__(
            self,
            hidden_dim: int,
            input_dim: int,
            output_dim: int,
            num_layers: int = 0,
            activation: str = "leaky_relu",
            output_activation: Optional[str] = "identity"
        ) -> None:
        super().__init__()

        if activation not in activation_table:
            raise ValueError(f"Unknown activation: {activation}")
        if output_activation not in activation_table:
            raise ValueError(f"Unknown output_activation: {output_activation}")
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.activation = activation_table[activation]()
        self.output_activation = activation_table[output_activation]()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        logits = self.output_layer(x)
        if self.output_activation:
            logits = self.output_activation(logits)

        return logits
    
