@echo "Train Prune"
for %%a in (resnet20, lenet5) do (
for %%m in (train_moo,baseline_moo) do (
for %%p in (0.1,0.2,0.3) do (
for %%s in (1,2,3,4) do (
    CALL python run.py %%m --model %%a --moo_num_sparse_classes %%s --moo_num_classes 5 --moo_sparse_ratio %%p
)
)
)
)
PAUSE 
