from abc import ABC, abstractmethod


class BaseEvolver(ABC):
    def __init__(self):
        self.korean_tail = "Answer in Korean."

    @abstractmethod
    def evolve(self, instruction, evolving_method):
        pass
