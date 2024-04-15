#/bin/bash

python train_4folds_freeze_AWP2.py --config /kaggle/scripts/yaml/EX026.yaml > /kaggle/output/EX026/log.txt
# kaggle datasets create -p /kaggle/output/EX023 --dir-mode zip > /kaggle/output/EX023/log2.txt
# kaggle datasets create -p /kaggle/output/EX024 --dir-mode zip > /kaggle/output/EX024/log2.txt
# kaggle datasets create -p /kaggle/output/EX025 --dir-mode zip > /kaggle/output/EX025/log2.txt
# kaggle datasets create -p /kaggle/output/EX026 --dir-mode zip > /kaggle/output/EX026/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813
