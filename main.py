import argparse
from dotenv import load_dotenv
from auto_evol import AutoEvolInstruct
import logging


def parse_key_value(item):
    key, value = item.split("=", 1)
    return {key: value}


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        type=str,
        default="beomi/KoAlpaca-v1.1a",
        help="Path to the hf dataset.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=20,
        help="Size of data to join method optimization.",
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=50,
        help="Size of data to join method validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for whole process.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Maximum number of steps for method optimization.",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=3,
        help="Whole generation of method evolving.(l in the paper)",
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=5,
        help="Number of optimizatied methods.(m in the paper)",
    )
    parser.add_argument(
        "--test-run",
        type=int,
        default=0,
        help="Test run or not.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="evolution_result.json",
        help="Path to save the evolution result.",
    )
    parser.add_argument(
        "--evol-llm-config",
        type=str,
        default="temperature=0 top_p=0 max_tokens=4096 timeout=120 max_retries=2",
        nargs="+",
        help="Configuration for the evolving LLM.",
    )
    parser.add_argument(
        "--optim-llm-config",
        type=str,
        default="temperature=0.6 top_p=0.95 max_tokens=4096 timeout=120 max_retries=2",
        nargs="+",
        help="Configuration for the optimizing LLM.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()

    # LLM Config 파싱
    config = get_config()
    evol_llm_config = {}
    for item in config.evol_llm_config.split():
        evol_llm_config.update(parse_key_value(item))
    optim_llm_config = {}
    for item in config.optim_llm_config.split():
        optim_llm_config.update(parse_key_value(item))
    config.evol_llm_config = evol_llm_config
    config.optim_llm_config = optim_llm_config

    # logging 설정
    logging.basicConfig(filename="evol_log.log", filemode="w", level=logging.INFO)

    # AutoEvolInstruct 실행
    auto_evol = AutoEvolInstruct(config)
    evolved_method = auto_evol.run_auto_evol(config.max_steps)
    auto_evol.run_evol_instruct(evolved_method, test_run=config.test_run)
