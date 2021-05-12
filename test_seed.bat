@echo "Train Prune"
for %%a in (resnet20,lenet5) do (
for %%m in (train_moo,baseline_moo) do (
    CALL python run.py %%m --model %%a --moo_num_sparse_classes 8 --moo_num_classes 10
)
)
PAUSE 
