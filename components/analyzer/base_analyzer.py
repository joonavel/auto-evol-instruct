from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    def __init__(self):
        self.korean_tail = "Answer in Korean."
    @abstractmethod
    def analyze(self, instruction, evolving_method):
        pass