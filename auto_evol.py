from utils.utils import CustomDataLoader, get_deepseek_llm
from datasets import Dataset
from components.evolver.instruction_evolver import InstructionEvolver
from components.analyzer.trajectory_analyzer import TrajectoryAnalyzer
from components.optimizer.method_optimizer import MethodOptimizer
from components.generator.response_generator import ResponseGenerator
from components.validator.response_validator import ResponseValidator

from typing import Tuple, List, Optional
import json
import logging

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
        self.save_path = config.save_path
        
    def load_data_for_auto_evol(self) -> Tuple[Dataset, Dataset]:
        train_dataset, dev_dataset = self.data_loader.get_train_and_dev()
        return train_dataset, dev_dataset
    
    def load_data_for_instruction_evolution(self) -> Dataset:
        instruction_dataset = self.data_loader.get_instruction_data()
        return instruction_dataset
    
    def evolve_instruction(self, train_dataset, is_initial=True, current_method: Optional[str] = None) -> List[List[str]]:
        evolver = InstructionEvolver(
            llm=get_deepseek_llm(temperature=0, max_tokens=4096, timeout=120, max_retries=2),
            dataset=train_dataset,
            ds_size=self.config.train_size,
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

    def validate_method(self, dev_dataset, candidate_methods: List[str]):
        # instruction evolving을 진행할 llm
        evolver = InstructionEvolver(
            llm=get_deepseek_llm(temperature=0, max_tokens=4096, timeout=120, max_retries=2),
            dataset=dev_dataset,
            ds_size=self.config.dev_size,
            loop=self.config.loop,
            batch_size=self.config.batch_size,
        )
        # 각 candidate method에 대해서 instruction evolving을 진행하고, 그 결과를 2차원 리스트로 저장
        dev_instructions_2d = []
        for idx, method in enumerate(candidate_methods):
            dev_output = evolver.evolve_once(method)
            logging.info(f"Dev Instructions has been evolved with {idx}th candidate method.")
            dev_instructions_2d.append(dev_output)
        
        # Instruction에 대한 답변을 생성할 llm
        responser = ResponseGenerator(
            llm=get_deepseek_llm(temperature=0, max_tokens=4096, timeout=120, max_retries=2),
            ds_size=self.config.dev_size,
            batch_size=self.config.batch_size,
        )
        validator = ResponseValidator(ds_size=self.config.dev_size)
        # 각 candidate method에 대해서 답변 생성을 진행
        scores = []
        for dev_instructions in dev_instructions_2d:
            responses = responser.generate(dev_instructions)
            # 답변 중 오류율을 validator를 이용해 계산 후 저장
            score = validator.validate(responses)
            scores.append(score)
        # 오류율이 최저인 candidate method를 선택하여 return
        return candidate_methods[scores.index(min(scores))]
    
    def save_evolution_result(self, parent_dataset, instructions: List[str], responses: List[str], save_path: str = "evolution_result.json"):
        with open(save_path, "w") as f:
            json.dump({"instruction": instructions,
                       "response": responses,
                       "root": parent_dataset["instruction"],
                       "reference": parent_dataset["root"]}, f)
        return
        
    def run_auto_evol(self, max_step):
        train_dataset, dev_dataset = self.load_data_for_auto_evol()
        for step in range(max_step):
            logging.info(f"Auto Evolution Step {step} has started.")
            # Instruction Evolution
            if step == 0:
                logging.info(f"Enter Initial Evolution\nCurrent Method : None")
                trajectory = self.evolve_instruction(train_dataset, is_initial=True)
            else:
                logging.info(f"Enter Iterative Evolution\nCurrent Method : {self.current_method}")
                trajectory = self.evolve_instruction(train_dataset, is_initial=False, current_method=self.current_method)
            # Trajectory Analysis
            result, feedback = self.analyze_trajectory(trajectory)
            logging.info(f"Step {step} : Trajectory Analysis Result: \n{result}")
            logging.info(f"Step {step} : Feedback : \n{feedback}\n")
            if result == "Error":
                logging.error("Trajectory Analysis Result is Error. Please check the trajectory.")
                return
            # Method Optimization
            if step == 0:
                candidate_methods = self.optimize_method(None, feedback, is_initial=True)
            else:
                candidate_methods = self.optimize_method(self.current_method, feedback, is_initial=False)
            if candidate_methods is None:
                logging.error("Candidate Methods are None. Please check the candidate size.")
                return
            # Method Selection
            best_method = self.validate_method(dev_dataset, candidate_methods)
            self.current_method = best_method
            logging.info(f"Step {step} : Best Method : \n{best_method}\n")
        return self.current_method
    
    def run_evol_instruct(self, method: str, test_run: bool = False):
        instruction_dataset = self.load_data_for_instruction_evolution()
        if test_run:
            flag = 10
        else:
            flag = None
        instructions = instruction_dataset[:flag]
        instruction_size = len(instructions)
        # Instruction Evolution
        evolver = InstructionEvolver(
            llm=get_deepseek_llm(temperature=0, max_tokens=4096, timeout=120, max_retries=2),
            dataset=instructions,
            ds_size=instruction_size,
            loop=None,
            batch_size=self.config.batch_size,
        )
        child_instructions = evolver.evolve_once(method)
        # Response Generation with Post-processing
        responser = ResponseGenerator(
            llm=get_deepseek_llm(temperature=0, max_tokens=4096, timeout=120, max_retries=2),
            ds_size=instruction_size,
            batch_size=self.config.batch_size,
        )
        responses = responser.generate_with_fix(child_instructions)
        # Saving
        self.save_evolution_result(instruction_dataset, child_instructions, responses, self.save_path)
        logging.info(f"Evolution Result has been saved.")
        return