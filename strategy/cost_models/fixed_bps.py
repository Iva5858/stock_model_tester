import numpy as np
from .base_cost_model import BaseCostModel, register_cost_model


@register_cost_model("fixed_bps")
class FixedBpsCostModel(BaseCostModel):
    def __init__(self, bps: float = 10.0):
        self.bps = bps

    def cost_per_step(self, position_before: float, position_after: float) -> float:
        if np.sign(position_before) != np.sign(position_after):
            return self.bps / 10000.0
        return 0.0
