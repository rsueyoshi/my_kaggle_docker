#/bin/bash

python train_4folds_freeze_AWP.py --config yaml/deberta3large_0966_bf16_4folds0_freeze_AWP.yaml > /kaggle/output/deberta3large_fold0_freeze6_AWP/log.txt
python train_4folds_freeze_AWP.py --config yaml/deberta3large_0966_bf16_4folds1_freeze_AWP.yaml > /kaggle/output/deberta3large_fold1_freeze6_AWP/log.txt
python train_4folds_freeze_AWP.py --config yaml/deberta3large_0966_bf16_4folds2_freeze_AWP.yaml > /kaggle/output/deberta3large_fold2_freeze6_AWP/log.txt
python train_4folds_freeze_AWP.py --config yaml/deberta3large_0966_bf16_4folds3_freeze_AWP.yaml > /kaggle/output/deberta3large_fold3_freeze6_AWP/log.txt
kaggle datasets create -p /kaggle/output/deberta3large_fold0_freeze6 --dir-mode zip
kaggle datasets create -p /kaggle/output/deberta3large_fold1_freeze6 --dir-mode zip
kaggle datasets create -p /kaggle/output/deberta3large_fold2_freeze6 --dir-mode zip
kaggle datasets create -p /kaggle/output/deberta3large_fold3_freeze6 --dir-mode zip
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813	
