import torch 
import torch.nn as nn

class ZeroTimeInflationPiProjector(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(config.vocab_size, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 1, bias=False)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = l(x)
        return x