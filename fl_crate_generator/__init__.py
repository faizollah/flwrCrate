"""fl_crate_generator: capture a Flower federated learning run and emit an RO-Crate."""

from .tracker import FLCrateTracker

__all__ = ["FLCrateTracker"]
__version__ = "0.2.0"