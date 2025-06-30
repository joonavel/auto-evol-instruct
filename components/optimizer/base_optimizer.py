from abc import ABC, abstractmethod
from typing import List, Optional


class BaseOptimizer(ABC):
    @abstractmethod
    def optimize(
        self, method: Optional[str], feedback: str, is_initial: bool = True
    ) -> Optional[List[str]]:
        pass
