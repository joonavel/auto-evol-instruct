from .base_evolver import BaseEvolver
from utils.utils import load_prompt_template

from pydantic import BaseModel, Field
from typing import List, Optional, Any
from langchain_core.prompts import ChatPromptTemplate
import logging
from tqdm import tqdm


# Pydantic models
class InitialEvolution(BaseModel):
    """Steps to make the instruction more complex"""
    methods : List[str] = Field(description="List of methods to make the instruction more complex")
    plan : List[str] = Field(description="Comprehensive plan based on the #Methods List# generated in Step 1 to make the #Instruction# more complex. The plan should include several methods from the #Methods List#.")
    rewritten_instruction : str = Field(description="Rewritten instruction that adds 10 to 20 words into the #Instruction#.")
    finally_rewritten_instruction : str = Field(description="Finally rewritten instruction after reviewing the #Rewritten Instruction# and identifying any unreasonable parts. Ensure that the #Rewritten Instruction# is only a more complex version of the #Instruction#, make sure that it only adds 10 to 20 words into the #Instruction#. Just provide without any explanation.")
    
class IterativeEvolution(BaseModel):
    """Finally rewritten instruction to give to user"""
    finally_rewritten_instruction : str = Field(description="Finally rewritten instruction after reviewing the #Rewritten Instruction# and identifying any unreasonable parts. Ensure that the #Rewritten Instruction# is only a more complex version of the #Instruction#, make sure that it only adds 10 to 20 words into the #Instruction#. Just provide without any explanation.")


class InstructionEvolver(BaseEvolver):
    def __init__(self, llm, dataset, ds_size, loop, batch_size):
        super().__init__()
        self.llm = llm
        self.dataset = dataset
        self.ds_size = ds_size
        self.loop = loop
        self.batch_size = batch_size
        self.system_prompt = load_prompt_template(
            "components/evolver/prompts/system_prompt.prompt"
        )
        self.initial_evolving_method = load_prompt_template(
            "components/evolver/prompts/initial_evolving_method.prompt"
        )
        self.iterative_evolving_method = load_prompt_template(
            "components/evolver/prompts/iterative_evolving_method.prompt"
        )
        
    def handle_parsing_error(self, instructions: List[str], batch_output: List[Any]) -> List[str]:
        """
        LLM이 출력한 output에서 Parsing error가 발생한 경우, 원본 instruction을 반환하고, 그렇지 않은 경우, 정상 출력을 반환

        Args:
            instructions (List[str]): evolving 대상 instruction
            batch_output (List[Any]): LLM이 출력한 evolving 결과 instructions

        Returns:
            List[str]: evolving 결과 instructions
        """
        temp = []
        for idx, output in enumerate(batch_output):
            if output['parsing_error']:
                temp.append(instructions[idx])
                logging.error(f"Parsing error at InstructionEvolution.\n{output['raw'].additional_kwargs['tool_calls'][0]['function']['arguments']}")
            else:
                temp.append(output['parsed'].finally_rewritten_instruction)
        return temp
    
    def _initial_evolve(self) -> List[List[str]]:
        evolver = self.llm.with_structured_output(InitialEvolution, include_raw=True)
        prompt = ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                ("user", self.initial_evolving_method + self.korean_tail),
            ]
        )
        chain = prompt | evolver
        # 답변을 모아둘 곳
        outputs = []
        # 배치 사이즈만큼 데이터를 나누어서 처리
        for k in tqdm(range(0, self.ds_size, self.batch_size), desc="Evolving..."):
            # 마지막 배치 처리
            if k + self.batch_size > self.ds_size:
                instructions = self.dataset['instruction'][k:]
            else:
                instructions = self.dataset['instruction'][k:k+self.batch_size]
            
            # 배치 입력 생성
            batch_input = [{"instruction": instruction} for instruction in instructions]
            
            # 배치 출력 저장소 초기화(배치 크기 만큼)
            batch_outputs = [[x] for x in instructions]
            
            try:
                # 하나의 method로 반복 evolving
                for _ in tqdm(range(self.loop), desc="In the batch..."):
                    batch_output = chain.batch(batch_input)
                    responses = self.handle_parsing_error(instructions, batch_output)
                    batch_input = [{"instruction": response} for response in responses]
                    for i, response in enumerate(responses):
                        batch_outputs[i].append(response)
            except Exception as e:
                logging.error(f"Batch {k}~{k+self.batch_size} failed: {e}")
                continue
            
            # 배치 출력 모으기
            outputs.extend(batch_outputs)
        logging.info(f"Evolution Loop output length: {len(outputs)}")
        return outputs
        
    def _iterative_evolve(self, current_method: str) -> List[List[str]]:
        evolver = self.llm.with_structured_output(IterativeEvolution)
        prompt = ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                ("user", self.iterative_evolving_method + self.korean_tail),
            ]
        )
        chain = prompt | evolver
        # 답변을 모아둘 곳
        outputs = []
        # 배치 사이즈만큼 데이터를 나누어서 처리
        for k in tqdm(range(0, self.ds_size, self.batch_size), desc="Evolving..."):
            # 마지막 배치 처리
            if k + self.batch_size > self.ds_size:
                instructions = self.dataset['instruction'][k:]
            else:
                instructions = self.dataset['instruction'][k:k+self.batch_size]
            
            # 배치 입력 생성
            batch_input = [{"steps": current_method, "instruction": instruction} for instruction in instructions]
            
            # 배치 출력 초기화(배치 크기 만큼)
            batch_outputs = [[x] for x in instructions]
            
            try:
                # 하나의 method로 반복 evolving
                for _ in tqdm(range(self.loop), desc="In the batch..."):
                    batch_output = chain.batch(batch_input)
                    responses = [output.finally_rewritten_instruction for output in batch_output]
                    batch_input = [{"steps": current_method, "instruction": response} for response in responses]
                    for i, response in enumerate(responses):
                        batch_outputs[i].append(response)
            except Exception as e:
                logging.error(f"Batch {k}~{k+self.batch_size} failed: {e}")
                continue
            
            # 배치 출력 모으기
            outputs.extend(batch_outputs)
        return outputs
                
    def evolve(self, is_initial: bool, current_method: Optional[str] = None) -> List[List[str]]:
        if is_initial:
            return self._initial_evolve()
        else:
            return self._iterative_evolve(current_method)
        
    def evolve_train(self, current_method: str):
        evolver = self.llm.with_structured_output(IterativeEvolution)
        prompt = ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                ("user", self.iterative_evolving_method + self.korean_tail),
            ]
        )
        chain = prompt | evolver
        
        outputs = []
        for k in tqdm(range(0, self.ds_size, self.batch_size)):
            if k + self.batch_size > self.ds_size:
                instructions = self.dataset['instruction'][k:]
            else:
                instructions = self.dataset['instruction'][k:k+self.batch_size]
            batch_input = [{"steps": current_method, "instruction": instruction} for instruction in instructions]
            batch_output = chain.batch(batch_input)
            responses = [output.finally_rewritten_instruction for output in batch_output]
            outputs.extend(responses)
        
        return outputs
