from .base_validator import BaseValidator
from typing import List

class ResponseValidator(BaseValidator):
    def __init__(self, ds_size):
        self.ds_size = ds_size
        self.FAILURE_RESPONSE = 'IDK'
        self.REPLACEMENT_CHAR = '\uFFFD'
        
    def validate(self, responses: List[str]) -> float:
        fail_count = 0
        for response in responses:
            if response == self.FAILURE_RESPONSE:
                fail_count += 1
            elif self.REPLACEMENT_CHAR in response:
                fail_count += 1
        return fail_count / self.ds_size
