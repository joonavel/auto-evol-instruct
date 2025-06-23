# auto-evol-instruct
Automatic Instruction Evolution for Commercial Synthetic Data

## Introduction
This repository contains developed version of an automated algorithm for augmenting Korean language data through instruction evolution(https://github.com/joonavel/evol-instruct), based on the Automatic Instruction Evolving technique from the following paper:

Automatic Instruction Evolving for Large Language Models
(https://arxiv.org/abs/2406.00770)

Previous version of the implementation has limited number of evolving methods so that, it was not suitable for certain types of data. ex) Mathmatical problems, Code generation, etc.

This implementation has been developed to solve this problem by adding automatic evolving method optimization.

## Installation
```
```

## Setting Environment Variables
Before running the project, you need to set the following environment variables in your .env file:
```
DEEPSEEK_API_KEY=your_deepseek_api_key
MY_HF_TOKEN=your_huggingface_api_key
```