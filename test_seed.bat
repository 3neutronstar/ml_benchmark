@echo "Train Prune"
for %%p in (0.15) do (
for %%s in (1) do (
for %%m in (baseline_moo,train_moo) do (
    CALL python run.py %%m --model resnet20 --moo_num_sparse_classes %%s --moo_num_classes 5 --moo_sparse_ratio %%p
)
)
)
PAUSE 
