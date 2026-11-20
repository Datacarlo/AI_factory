"""Load a trained price model together with its input order and normalization."""
from pathlib import Path
import torch
from learning.common.networks import build_network


class FastPricer(torch.nn.Module):
    def __init__(self, checkpoint: str | Path, *, device='cpu'):
        super().__init__()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['representation']['name'] != 'identity':
            raise ValueError('This reader supports the catalogue identity representation')
        self.input_policy = saved['input_policy']
        self.feature_names = tuple(saved['tensor_schema']['feature_names'])
        self.network = build_network(saved['network'], input_dimension=len(self.feature_names), output_dimension=1)
        self.network.load_state_dict(saved['model_state_dict'])
        for name,value in saved['transform'].items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
        self.to(device).eval()

    def forward(self, inputs):
        """Contract coordinates in feature_names order, times in years; output is price."""
        if inputs.shape[-1] != len(self.feature_names):
            raise ValueError('Unexpected feature dimension')
        normalized = (inputs-self.feature_mean)/self.feature_scale
        return self.network(normalized)*self.value_scale+self.value_mean
