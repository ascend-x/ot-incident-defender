from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from environment.models import Observation, Action


class BasePlant(ABC):

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._step_count: int = 0

    @abstractmethod
    def reset(self) -> Observation:
        ...

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        ...

    @property
    def current_step(self) -> int:
        return self._step_count
