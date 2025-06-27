from datasets import load_dataset, Dataset
from langchain_deepseek import ChatDeepSeek
from typing import Tuple

class CustomDataLoader:
    """
    데이터셋을 로드하고, 셔플하며, train/dev 셋으로 분리하는 역할을 담당합니다.
    """
    def __init__(self, data_path: str, train_size: int, dev_size: int, seed: int):
        """
        DataLoader를 초기화합니다.

        Args:
            data_path (str): 로드할 데이터셋의 경로 또는 Hugging Face Hub 이름.
            train_size (int): 훈련 세트의 크기.
            dev_size (int): 개발 세트의 크기.
            seed (int): 셔플에 사용할 시드 값.
        """
        self.data_path = data_path
        self.train_size = train_size
        self.dev_size = dev_size
        self.seed = seed
        
    def get_train_and_dev(self) -> Tuple[Dataset, Dataset]:
        """
        데이터셋을 로드하고, 셔플하며, train/dev 셋으로 분리합니다.

        Raises:
            ValueError: 훈련 세트와 개발 세트의 크기가 데이터셋의 크기보다 큰 경우 예외를 발생시킵니다.

        Returns:
            tuple: (train_dataset, dev_dataset)
        """
        print(f"Loading dataset from {self.data_path}...")
        dataset = load_dataset(self.data_path, split="train")
        dataset = dataset.shuffle(seed=self.seed)
        train_ds = dataset.select(range(self.train_size))
        if self.train_size + self.dev_size > len(dataset):
            raise ValueError("Train and dev size is too large for the dataset.\n"
                             f"Dataset size: {len(dataset)}")
        dev_ds = dataset.select(range(self.train_size, self.train_size + self.dev_size))
        return train_ds, dev_ds
    
    def get_instruction_data(self) -> Dataset:
        """
        최적화된 method로 evolving 하기 위한 데이터셋을 준비합니다.
        """
        print(f"Loading dataset from {self.data_path}...")
        dataset = load_dataset(self.data_path, split="train")
        return dataset
    
def load_prompt_template(prompt_template_path: str) -> str:
    """
    Prompt template을 로드합니다.

    Args:
        prompt_template_path (str): Prompt template의 경로

    Returns:
        str: Prompt template
    """
    with open(prompt_template_path, "r") as f:
        return f.read()
    

def get_deepseek_llm(max_tokens: int, timeout: int, max_retries: int, temperature: float, top_p: float | None = None,) -> ChatDeepSeek:
    """
    DeepSeek LLM을 초기화합니다.
    """
    if temperature == 0:
        print(f"Loading DeepSeek LLM for evolution...")
        llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature = 0,
            max_tokens = max_tokens,
            timeout = timeout,
            max_retries = max_retries,
        )
    else:
        print(f"Loading DeepSeek LLM for optimization...")
        llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature = temperature,
            top_p = top_p,
            max_tokens = max_tokens,
            timeout = timeout,
            max_retries = max_retries,
        )
    return llm