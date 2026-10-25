from types import SimpleNamespace

import torch
from torch import nn

from learning.common.training.losses import build_composite_loss
from learning.deep_pricing.representations import UnitSpotSliceModel


def test_gradient_loss_weights_named_coordinates_and_ignores_zero_weight():
    loss = build_composite_loss([
        {"name": "gradient_mse", "weight": 0.1,
         "coordinate_weights": {"model.spot": 0.0, "product.strike": 2.0}},
    ])
    context = SimpleNamespace(
        normalized_predicted_gradients=torch.tensor([[100.0, 2.0]]),
        normalized_target_gradients=torch.tensor([[0.0, 0.0]]),
        gradient_names=("model.spot", "product.strike"),
        normalized_predictions=torch.zeros(1, 1),
    )
    value, parts = loss(context)
    assert torch.isclose(value, torch.tensor(0.4))
    assert torch.isclose(torch.tensor(parts["gradient_mse"]), torch.tensor(4.0))


def test_unit_spot_slice_model_does_not_use_spot_coordinate():
    core = nn.Linear(2, 1, bias=False)
    model = UnitSpotSliceModel(core, spot_index=1)
    inputs = torch.tensor([[0.2, 1.0, -0.5]], requires_grad=True)
    prediction = model(inputs)
    gradient = torch.autograd.grad(prediction.sum(), inputs)[0]
    changed_spot = inputs.detach().clone()
    changed_spot[:, 1] = 17.0
    assert torch.equal(prediction.detach(), model(changed_spot))
    assert gradient[0, 1].item() == 0.0
