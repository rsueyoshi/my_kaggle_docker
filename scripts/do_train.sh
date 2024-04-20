#/bin/bash

python train_4folds_freeze_AWP22.py --config /kaggle/scripts/yaml/EX035.yaml > /kaggle/output/EX035/log.txt
python train_4folds_freeze_AWP2.py --config /kaggle/scripts/yaml/EX031.yaml > /kaggle/output/EX031/log.txt
# kaggle datasets create -p /kaggle/output/EX034 --dir-mode zip > /kaggle/output/EX034/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813
