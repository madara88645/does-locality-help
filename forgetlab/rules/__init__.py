"""Learning rules. Each returns the weight changes to add, given a batch."""

from forgetlab.rules.backprop import backprop_updates
from forgetlab.rules.chl import chl_updates
from forgetlab.rules.predictive_coding import pc_updates

RULES = {
    "backprop": backprop_updates,
    "pc": pc_updates,
    "chl": chl_updates,
}

__all__ = ["RULES", "backprop_updates", "chl_updates", "pc_updates"]
