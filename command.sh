echo 'train lenet5'
for VALUE in ("50","75","100","125","150")
CUDA_VISIBLE_DEVICES=1 python run.py train --threshold $VALUE
python run.py 