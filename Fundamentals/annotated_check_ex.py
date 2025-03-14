from typing import Annotated, TypeVar, Protocol
import numpy as np
import torch

# Define a protocol ensuring objects have a __len__ method
class HasLength(Protocol):
    def __len__(self) -> int: ...

# Generic TypeVar bound to types with __len__
T = TypeVar("T", bound=HasLength)

# Length constraint metadata class
class LengthConstraint:
    def __init__(self, min_length: int = 0, max_length: int | float = float("inf")) -> None:
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, value: HasLength) -> None:
        if not (self.min_length <= len(value) <= self.max_length):
            raise ValueError(f"Value length must be between {self.min_length} and {self.max_length}, got {len(value)}")

# Define a generic annotation using Annotated
TensorMatch = Annotated[T, LengthConstraint]

# Function to test `mypy` with Annotated types
def check_tensor(tensor: TensorMatch) -> None:
    if not isinstance(tensor, (np.ndarray, torch.Tensor)):
        raise TypeError(f"Expected a tensor-like object, got {type(tensor).__name__}")

    constraint = LengthConstraint(3, 10)  # Example constraint
    constraint.validate(tensor)

    print(f"Valid tensor with shape: {tensor.shape}")

# Example usage (Valid case)
valid_tensor = np.random.rand(5, 5)  # Length is 5 (valid)
check_tensor(valid_tensor)

# Uncomment this line to see `mypy` detect a type issue:
# check_tensor(123)  # Invalid type (should raise a mypy error)

