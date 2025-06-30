from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    def __init__(self):
        self.korean_tail = "Answer in Korean."

    @abstractmethod
    def generate(self, instructions: str) -> str:
        pass
