@echo "Seed Change"

for %%a in (1,1,35) do (
    CALL python run.py train --test_seed %%a
)

PAUSE