from utils.utils import CustomDataLoader, get_deepseek_llm
from datasets import Dataset
from components.evolver.instruction_evolver import InstructionEvolver
from components.analyzer.trajectory_analyzer import TrajectoryAnalyzer
from components.optimizer.method_optimizer import MethodOptimizer

from typing import Tuple, List, Optional
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
        self.current_method = None
        
    def load_data_for_auto_evol(self) -> Tuple[Dataset, Dataset]:
        train_dataset, dev_dataset = self.data_loader.get_train_and_dev()
        return train_dataset, dev_dataset
    
    def load_data_for_instruction_evolution(self) -> Dataset:
        instruction_dataset = self.data_loader.get_instruction_data()
        return instruction_dataset
    
    def evolve_instruction(self, train_dataset, is_initial=True, current_method: Optional[str] = None) -> List[List[str]]:
        evolver = InstructionEvolver(
            llm=get_deepseek_llm(temperature=0, max_tokens=4096, timeout=120, max_retries=2),
            train_dataset=train_dataset,
            train_size=self.config.train_size,
            loop=self.config.loop,
            batch_size=self.config.batch_size,
        )
        return evolver.evolve(is_initial=is_initial, current_method=current_method)

    def analyze_trajectory(self, trajectory : List[List[str]]) -> Tuple[str, str | None]:
        analyzer = TrajectoryAnalyzer(
            llm=get_deepseek_llm(temperature=0.6, top_p=0.95, max_tokens=4096, timeout=120, max_retries=2),
        )
        return analyzer.analyze(trajectory)
    
    def optimize_method(self, method: str, feedback: str, is_initial: bool = True) -> List[str] | None:
        optimizer = MethodOptimizer(
            llm=get_deepseek_llm(temperature=0.6, top_p=0.95, max_tokens=4096, timeout=120, max_retries=2),
            candidate_size=self.config.candidate_size,
        )
        return optimizer.optimize(method, feedback, is_initial=is_initial)

    def run(self, max_step):
        train_dataset, dev_dataset = self.load_data_for_auto_evol()
        for step in range(max_step):
            # Instruction Evolution
            if step == 0:
                trajectory = self.evolve_instruction(train_dataset, is_initial=True)
            else:
                trajectory = self.evolve_instruction(train_dataset, is_initial=False)
            # Trajectory Analysis
            result, feedback = self.analyze_trajectory(trajectory)
            print(result)
            print(feedback)
            if result == "Error":
                return
            # Method Optimization
            if step == 0:
                candidate_methods = self.optimize_method(None, feedback, is_initial=True)
            else:
                candidate_methods = self.optimize_method(self.current_method, feedback, is_initial=False)
            print(candidate_methods)
            if candidate_methods is None:
                return
            # Method Selection
            
            
            
        
        
        

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
