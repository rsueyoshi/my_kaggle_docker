#/bin/bash

python train_alldata.py --config yaml/deberta3large_0966_bf16_all_freeze.yaml > /kaggle/output/deberta3large_freeze6/log.txt
kaggle datasets create -p /kaggle/output/deberta3large_freeze6 --dir-mode zip
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813	
