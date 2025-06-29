#! /bin/bash

python main.py\
    --data-path joonavel/seed_for_evolving\
    --train-size 20\
    --dev-size 50\
    --seed 42\
    --batch-size 10\
    --max-step 3\
    --loop 3\
    --candidate-size 5\
    --test-run 0\
    --save-path evolution_result.json
