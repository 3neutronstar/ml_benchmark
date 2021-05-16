@echo "Train Prune"
for %%a in (resnet20) do (
for %%m in (train_moo,baseline_moo) do (
for %%p in (0.2,0.3,0.4) do (
for %%s in (1,2,3,4,5) do (
    CALL python run.py %%m --model %%a --moo_num_sparse_classes %%s --moo_num_classes 10 --moo_sparse_ratio %%p
)
)
)
)
PAUSE 
