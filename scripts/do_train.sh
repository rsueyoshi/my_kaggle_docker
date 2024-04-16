#/bin/bash

# python train_4folds_freeze_AWP2.py --config /kaggle/scripts/yaml/EX027.yaml > /kaggle/output/EX027/log.txt
python train_4folds_freeze_AWP2.py --config /kaggle/scripts/yaml/EX029.yaml > /kaggle/output/EX029/log.txt
python train_4folds_freeze_AWP2.py --config /kaggle/scripts/yaml/EX028.yaml > /kaggle/output/EX028/log.txt
# kaggle datasets create -p /kaggle/output/EX027 --dir-mode zip > /kaggle/output/EX027/log2.txt
# kaggle datasets create -p /kaggle/output/EX029 --dir-mode zip > /kaggle/output/EX029/log2.txt
# kaggle datasets create -p /kaggle/output/EX028 --dir-mode zip > /kaggle/output/EX028/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813
