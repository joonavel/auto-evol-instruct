#! /bin/bash

python main.py\
    --is-local-data 1\
    --data-path korean_culture_seed.json\
    --instruction-field question\
    --train-size 10\
    --dev-size 20\
    --seed 42\
    --batch-size 10\
    --max-step 3\
    --loop 3\
    --candidate-size 5\
    --test-run 1\
    --save-path evolution_result.json
