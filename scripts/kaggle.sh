#/bin/bash

kaggle datasets create -p /kaggle/output/EX026 --dir-mode zip > /kaggle/output/EX026/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813
