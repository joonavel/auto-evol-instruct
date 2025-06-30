from .base_optimizer import BaseOptimizer
from utils.utils import load_prompt_template

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import logging
from typing import List, Optional


class MethodOptimization(BaseModel):
    """Optimized method for evolving the instructions."""

    optimized_method: str = Field(
        description="Optimized method for evolving the instructions."
    )


initial_method_template = """Step 1: Please read the "#Instruction#" below carefully and list all the possible methods to make this instruction more complex (to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). Please do not provide methods to change the language of the instruction!

Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the #Instruction# more complex. The plan should include several methods from the #Methods List#.

Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only add 10 to 20 words into the "#Instruction#".

Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. Ensure that the #Rewritten Instruction# is only a more complex version of the #Instruction#, make sure that it only adds 10 to 20 words into the "#Instruction#". Just provide the #Finally Rewritten Instruction# without any explanation.
"""


class MethodOptimizer(BaseOptimizer):
    def __init__(self, llm, candidate_size: int):
        self.llm = llm
        self.candidate_size = candidate_size
        self.initial_method_template = initial_method_template
        self.system_prompt = load_prompt_template(
            "components/optimizer/prompts/system_prompt.prompt"
        )
        self.method_optimization_prompt = load_prompt_template(
            "components/optimizer/prompts/method_optimization_prompt.prompt"
        )

    def optimize_method(
        self, method: Optional[str], feedback: str, is_initial: bool = True
    ) -> Optional[List[str]]:
        if self.candidate_size <= 0:
            logging.error("Candidate size must be greater than 0.")
            return None

        optimizer = self.llm.with_structured_output(MethodOptimization)
        prompt = ChatPromptTemplate(
            [("system", self.system_prompt), ("user", self.method_optimization_prompt)]
        )
        chain = prompt | optimizer
        if is_initial:
            method = self.initial_method_template
        print("Optimization has started...")
        try:
            opt_result = chain.batch(
                [{"current_method": method, "feedback": feedback}] * self.candidate_size
            )
            candidate_methods = [result.optimized_method for result in opt_result]
            return candidate_methods
        except Exception as e:
            logging.error(f"Error in optimizing method: {e}")
            return None

    def optimize(
        self, method: Optional[str], feedback: str, is_initial: bool = True
    ) -> Optional[List[str]]:
        return self.optimize_method(method, feedback, is_initial=is_initial)
