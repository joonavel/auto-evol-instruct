from utils.utils import CustomDataLoader
from datasets import Dataset
from typing import Tuple

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

# TODO
# 1. 데이터 준비

## 데이터 불러오기

## Train, Dev 분리

# 2. Auto Evolving for Method

## initial evolution

## trajectory analysis

## response generation

## validation

## repeat for set number of times

# 3. Instruction Evolution

## Evolving

## Answer Generation

## Post-processing

# 4. 결과 저장
