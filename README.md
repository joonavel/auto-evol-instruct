# auto-evol-instruct
Automatic Instruction Evolution is an framework designed to automatically enhance the complexity of instructions used for LLM fine-tuning without human effort.

## Keywords
- Method: Prompt to enhance the complexity of instructions
- Trajectory: Every stages of instruction evolved by the method

## Key Features
- Automatic instruction complexity enhancement without human effort
- Support for various types of data
- Flexible configuration for method optimization process

## Introduction
This repository contains developed version of an automated algorithm for augmenting Korean language data through instruction evolution(https://github.com/joonavel/evol-instruct), based on the Automatic Instruction Evolving technique from the following paper:

Automatic Instruction Evolving for Large Language Models
(https://arxiv.org/abs/2406.00770)

Previous version of the implementation has limited number of evolving methods so that, it was not suitable for certain types of data. ex) Mathmatical problems, Code generation, etc.

This implementation has been developed to solve this problem by adding automatic evolving method optimization.

## Installation
```
pip install -r requirements.txt
```

## Usage

### Setting Environment Variables
Before running the project, you need to set the following environment variables in your .env file:
```
DEEPSEEK_API_KEY=your_deepseek_api_key
MY_HF_TOKEN=your_huggingface_api_key
```

### Required Parameters
- ```data_path <hf_dataset_name>```: Path to the hf dataset. There should be "instruction" and "response" fields in the dataset.
- ```train_size <int>```: Size of data to join method optimization.
- ```dev_size <int>```: Size of data to join method validation.
- ```seed <int>```: Random seed for reproducibility.
- ```batch_size <int>```: Batch size for whole process.
- ```max_steps <int>```: Maximum number of steps for method optimization.
- ```loop <int>```: Whole generation of method evolving.(l in the paper)
- ```candidate_size <int>```: Number of optimizatied methods.(m in the paper)
- ```test_run <int>```: Test run or not.

### Optional Parameters
- ```evol_llm_config <str>```: Configuration for the evolving LLM.
- ```optim_llm_config <str>```: Configuration for the optimizing LLM.

### Model

### Example

## Components

## Output