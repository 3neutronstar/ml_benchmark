@echo "Seed Change"

for /l %%a in (1,1,35) do (
    CALL python run.py train --test_seed %%a
)

PAUSE 
