# Define needed Functions
import numpy as np
from jax import Array
import numpy.typing as npt
from flex.gp.util import compile_individuals

# Safe divide function
def safe_divide(a, b, default=1):
    """
    Perform a safe division. If `b` is zero or close to zero, return the `default` value.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) > 1e-9, a / b, default)
    return result


def safe_divideSR(a, b, default=1):
    """
    Perform a safe division. If `b` is zero or close to zero, return the `default` value.
    """
    return a / b if abs(b) > 1e-9 else default


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
