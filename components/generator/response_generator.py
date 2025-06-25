from .base_generator import BaseGenerator
from utils.utils import load_prompt_template

from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pydantic import BaseModel, Field
from tqdm import tqdm

class ResponseResult(BaseModel):
    """Answer to the user."""

    response: str = Field(description="Answer from the given question. If you don't know answer, answer with IDK.")

class ResponseGenerator(BaseGenerator):
    def __init__(self, llm, ds_size, batch_size):
        super().__init__()
        self.llm = llm
        self.ds_size = ds_size
        self.batch_size = batch_size
        self.system_prompt = load_prompt_template("components/generator/prompts/system_prompt.prompt")
        self.response_prompt = load_prompt_template("components/generator/prompts/response_prompt.prompt")
        
    def generate_response(self, instructions: str) -> str:
        responser = self.llm.with_structured_output(ResponseResult)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", self.response_prompt + self.korean_tail),
        ])
        chain = prompt | responser
        responses = []
        for k in tqdm(range(0, self.ds_size, self.batch_size)):
            if k + self.batch_size > self.ds_size:
                questions = instructions[k:]
            else:
                questions = instructions[k:k+self.batch_size]
            batch_input = [{"question": question} for question in questions]
            batch_output = chain.batch(batch_input)
            responses.extend([output.response for output in batch_output])
            
        return responses
    
    def generate(self, instructions: List[str]) -> List[str]:
        return self.generate_response(instructions)