#/bin/bash

python /kaggle/scripts/train_4folds_freeze.py --config /kaggle/scripts/yaml/EX005.yaml > /kaggle/output/EX005/log.txt
python /kaggle/scripts/train_4folds_freeze.py --config /kaggle/scripts/yaml/EX006.yaml > /kaggle/output/EX006/log.txt
python /kaggle/scripts/train_4folds_freeze.py --config /kaggle/scripts/yaml/EX007.yaml > /kaggle/output/EX007/log.txt
kaggle datasets create -p /kaggle/output/EX005 --dir-mode zip > /kaggle/output/EX005/log2.txt
kaggle datasets create -p /kaggle/output/EX006 --dir-mode zip > /kaggle/output/EX006/log2.txt
kaggle datasets create -p /kaggle/output/EX007 --dir-mode zip > /kaggle/output/EX007/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813	
