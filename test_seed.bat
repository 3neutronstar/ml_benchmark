@echo "Train Prune"
for %%a in (resnet20) do (
for %%p in (0.1,0.2,0.3,0.4) do (
for %%s in (1,2,3) do (
for %%m in (train_lbl,baseline_moo) do (
    CALL python run.py %%m --model %%a --moo_num_sparse_classes %%s --moo_num_classes 5 --moo_sparse_ratio %%p
)
)
)
)
PAUSE 
