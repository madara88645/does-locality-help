"""Learning rules. Each returns the weight changes to add, given a batch."""

from forgetlab.rules.backprop import backprop_updates
from forgetlab.rules.chl import chl_updates

__all__ = ["backprop_updates", "chl_updates"]
