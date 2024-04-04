from typing import Any
from typing import Tuple
from typing import TypeVar
from typing import Union

from params import SlotAttentionParams

import torch

def mask_renormalize(probs, mask):
  """Renormalizes probs with a mask so that the unmasked entries sum to 1.

  Args:
    probs (torch.FloatTensor): batch of probability distributions of shape
      (batch_dim1, batch_dim2, ..., num_elems).
    mask (torch.ByteTensor): mask of same shape as probs, where entries = 0
      should be masked out.

  Returns:
    renormalized_probs (torch.FloatTensor): tensor of same shape as probs. Each
      batch row (last dim) sums to 1, where masked entries have 0 prob. If all
      entries in a batch are masked, the batch row sums to 0.
  """
  mask=mask.to(probs.device)
  masked_probs = probs * mask
  renormalized = masked_probs / (masked_probs.sum(-1, keepdim=True) + 1e-8)
  return renormalized

Tensor = TypeVar("torch.tensor")
T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")
params = SlotAttentionParams()

def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"

def rescale(x: Tensor) -> Tensor:
    return x * 2 - 1 # type: ignore

def compact(l: Any) -> Any:
    return list(filter(None, l))


def first(x):
    return next(iter(x))


def only(x):
    materialized_x = list(x)
    assert len(materialized_x) == 1
    return materialized_x[0]


def to_rgb_from_tensor(x: Tensor):
    return (x * 0.5 + 0.5).clamp(0, 1) # type: ignore
