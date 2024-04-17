#/bin/bash

python train_4folds_freeze_AWP2.py --config /kaggle/scripts/yaml/EX028.yaml > /kaggle/output/EX028/log.txt
python train_4folds_freeze_AWP2.py --config /kaggle/scripts/yaml/EX030.yaml > /kaggle/output/EX030/log.txt
python train_4folds_freeze_AWP2.py --config /kaggle/scripts/yaml/EX031.yaml > /kaggle/output/EX031/log.txt
# kaggle datasets create -p /kaggle/output/EX031 --dir-mode zip > /kaggle/output/EX031/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813
