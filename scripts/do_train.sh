#/bin/bash

# python train_4folds_freeze_AWP22.py --config /kaggle/scripts/yaml/EX035.yaml > /kaggle/output/EX035/log.txt
# python train_4folds_freeze_AWP2.py --config /kaggle/scripts/yaml/EX038.yaml > /kaggle/output/EX038/log.txt
python train_alldata_freeze_AWP2.py --config /kaggle/scripts/yaml/EX037.yaml > /kaggle/output/EX037/log.txt
# kaggle datasets create -p /kaggle/output/EX036 --dir-mode zip > /kaggle/output/EX036/log2.txt
# kaggle datasets create -p /kaggle/output/EX038 --dir-mode zip > /kaggle/output/EX038/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813
