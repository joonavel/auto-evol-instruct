from abc import ABC, abstractmethod
from typing import List


class BaseValidator(ABC):
    @abstractmethod
    def validate(self, responses: List[str]) -> float:
        pass
