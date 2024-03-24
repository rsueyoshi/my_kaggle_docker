#/bin/bash

python train_alldata.py --config yaml/deberta3large_0966.yaml 
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813	
