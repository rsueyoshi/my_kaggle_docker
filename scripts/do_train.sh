#/bin/bash

python train_4folds_freeze_resume.py --config /kaggle/scripts/yaml/EX011.yaml > /kaggle/output/EX011/log.txt
python train_4folds_freeze_resume.py --config /kaggle/scripts/yaml/EX012.yaml > /kaggle/output/EX012/log.txt
python train_4folds_freeze.py --config /kaggle/scripts/yaml/EX013.yaml > /kaggle/output/EX013/log.txt
# python train_alldata.py --config /kaggle/scripts/yaml/EX010.yaml > /kaggle/output/EX010/log.txt
# kaggle datasets create -p /kaggle/output/EX010 --dir-mode zip > /kaggle/output/EX010/log2.txt
kaggle datasets create -p /kaggle/output/EX011 --dir-mode zip > /kaggle/output/EX011/log2.txt
kaggle datasets create -p /kaggle/output/EX010 --dir-mode zip > /kaggle/output/EX012/log2.txt
kaggle datasets create -p /kaggle/output/EX010 --dir-mode zip > /kaggle/output/EX013/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813
