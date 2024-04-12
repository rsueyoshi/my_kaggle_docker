#/bin/bash

python train_4folds_freeze_resume.py --config /kaggle/scripts/yaml/EX014.yaml > /kaggle/output/EX014/log.txt
python train_4folds_freeze.py --config /kaggle/scripts/yaml/EX015.yaml > /kaggle/output/EX015/log.txt
kaggle datasets create -p /kaggle/output/EX014 --dir-mode zip > /kaggle/output/EX014/log2.txt
kaggle datasets create -p /kaggle/output/EX015 --dir-mode zip > /kaggle/output/EX015/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813
