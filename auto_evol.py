from utils.utils import CustomDataLoader, get_deepseek_llm
from datasets import Dataset
from components.evolver.instruction_evolver import InstructionEvolver

from typing import Tuple, List
import json

class AutoEvolInstruct:
    def __init__(self, config):
        self.config = config
        self.data_loader = CustomDataLoader(
            data_path=config.data_path,
            train_size=config.train_size,
            dev_size=config.dev_size,
            seed=config.seed,
        )
        
    def load_data_for_auto_evol(self) -> Tuple[Dataset, Dataset]:
        train_dataset, dev_dataset = self.data_loader.get_train_and_dev()
        return train_dataset, dev_dataset
    
    def load_data_for_instruction_evolution(self) -> Dataset:
        instruction_dataset = self.data_loader.get_instruction_data()
        return instruction_dataset
    
    def evolve_instruction(self, train_dataset, is_initial=True) -> List[List[str]]:
        evolver = InstructionEvolver(
            llm=get_deepseek_llm(temperature=0, max_tokens=4096, timeout=120, max_retries=2),
            train_dataset=train_dataset,
            train_size=self.config.train_size,
            loop=self.config.loop,
            batch_size=self.config.batch_size,
        )
        if is_initial:
            return evolver._initial_evolve()
        else:
            return evolver._iterative_evolve()

    def run(self, max_step):
        train_dataset, dev_dataset = self.load_data_for_auto_evol()
        for step in range(max_step):
            if step == 0:
                trajectory = self.evolve_instruction(train_dataset, is_initial=True)
            else:
                trajectory = self.evolve_instruction(train_dataset, is_initial=False)
        
        result = trajectory
        # 결과를 파일로 작성
        with open(f"temp_result.json", "w") as f:
            json.dump(result, f)
        

# 1. 데이터 준비

## 데이터 불러오기

## Train, Dev 분리

# 2. Auto Evolving for Method

## initial evolution
# TODO
## trajectory analysis

## response generation

## validation

## repeat for set number of times

# 3. Instruction Evolution

## Evolving

## Answer Generation

## Post-processing

# 4. 결과 저장
