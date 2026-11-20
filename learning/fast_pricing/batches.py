"""Transfer one prepared dataset once; share tensors across all its runs."""
from __future__ import annotations
import numpy as np
import torch


class ResidentDataset:
    def __init__(self, prepared, device):
        self.device = device
        self.arrays = {name: torch.as_tensor(np.array(getattr(prepared, name), copy=True), device=device)
                      for name in ('features', 'values', 'gradients', 'diagonal_hessians',
                                   'value_standard_errors', 'gradient_standard_errors',
                                   'diagonal_hessian_standard_errors', 'entity_ordinals')}

    def loader(self, indices, *, batch_size, shuffle=False, seed=0, generator=None):
        return ResidentLoader(self, indices, batch_size, shuffle, seed, generator)


class ResidentLoader:
    def __init__(self, dataset, indices, batch_size, shuffle, seed, generator):
        self.dataset = dataset
        self.indices = torch.as_tensor(np.array(indices, copy=True), device=dataset.device)
        self.batch_size, self.shuffle, self.seed, self.generator = batch_size, shuffle, seed, generator

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = self.indices
        if self.shuffle:
            generator = self.generator
            if generator is None:
                generator = torch.Generator(device=self.dataset.device).manual_seed(self.seed)
            indices = indices[torch.randperm(len(indices), generator=generator, device=indices.device)]
        for selected in indices.split(self.batch_size):
            yield dict(index=selected, **{k: v[selected] for k, v in self.dataset.arrays.items()})
