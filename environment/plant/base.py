"""
base.py — Abstract base class for all plant physics simulators.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from environment.models import Observation, Action


class BasePlant(ABC):
    """
    Every scenario plant must inherit from BasePlant.
    Accepts a seed for deterministic behaviour and exposes reset() / step().
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._step_count: int = 0

    @abstractmethod
    def reset(self) -> Observation:
        """Reset plant to initial state and return first observation."""
        ...

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Apply action, advance physics, return (obs, reward_hint, done, info).
        reward_hint is a partial signal for the env wrapper to combine with the
        full reward computation.
        """
        ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return full internal state dict (may include hidden variables)."""
        ...

    @property
    def current_step(self) -> int:
        return self._step_count
