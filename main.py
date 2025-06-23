import os, argparse, sys
from dotenv import load_dotenv
from auto_evol import AutoEvolInstruct

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
        default=10,
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
    
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    
    config = get_config()
    auto_evol = AutoEvolInstruct(config)
    print(auto_evol.config)