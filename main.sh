#! /bin/bash

python main.py\
    --data-path beomi/KoAlpaca-v1.1a\
    --train-size 10\
    --dev-size 10\
    --seed 42\
    --batch-size 10\
    --max-step 2\
    --loop 3\
    --candidate-size 5\
    --test-run 1
