#/bin/bash

python train_alldata.py --config /kaggle/scripts/yaml/EX008.yaml > /kaggle/output/EX008/log.txt
python train_alldata.py --config /kaggle/scripts/yaml/EX009.yaml > /kaggle/output/EX009/log.txt
kaggle datasets create -p /kaggle/output/EX008 --dir-mode zip > /kaggle/output/EX008/log2.txt
kaggle datasets create -p /kaggle/output/EX009 --dir-mode zip > /kaggle/output/EX009/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813	
