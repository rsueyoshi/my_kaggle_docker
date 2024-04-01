#/bin/bash

python train_4folds_freeze.py --config yaml/deberta3large_0966_bf16_4folds1_freeze.yaml > /kaggle/output/deberta3large_fold1_freeze6/log.txt
python train_4folds_freeze.py --config yaml/deberta3large_0966_bf16_4folds2_freeze.yaml > /kaggle/output/deberta3large_fold2_freeze6/log.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813	
