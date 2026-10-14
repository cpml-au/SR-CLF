"""The Dataset container that flex expects as training data.

Dataset: a named list of input arrays; the runner (src/CPU/run.py) puts the axes of the training grid in it.
Importing this module needs jax (the Array type hint) and flex.
"""
from jax import Array
import numpy.typing as npt
from flex.gp.util import compile_individuals

# Dataset Function
class Dataset:
    def __init__(
        self,
        name: str,
        X: list[Array | npt.NDArray],
        y: Array | npt.NDArray | None = None,
    ) -> None:
        self.name = name
        self.X = X  # List of input arrays, e.g., [X1, X2]
        self.y = y

    def get_input(self, index: int) -> Array | npt.NDArray:
        """Get a specific input feature by index."""
        return self.X[index]
