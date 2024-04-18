#/bin/bash

python train_4folds_freeze_AWP2.py --config /kaggle/scripts/yaml/EX032.yaml > /kaggle/output/EX032/log.txt
python train_4folds_freeze_AWP2.py --config /kaggle/scripts/yaml/EX033.yaml > /kaggle/output/EX033/log.txt
python train_4folds_freeze_AWP2.py --config /kaggle/scripts/yaml/EX034.yaml > /kaggle/output/EX034/log.txt
python train_4folds_freeze_AWP2.py --config /kaggle/scripts/yaml/EX031.yaml > /kaggle/output/EX031/log.txt
# kaggle datasets create -p /kaggle/output/EX032 --dir-mode zip > /kaggle/output/EX032/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813
