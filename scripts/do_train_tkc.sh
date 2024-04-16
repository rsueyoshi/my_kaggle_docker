#/bin/bash

python train_4folds_freeze.py --config /kaggle/scripts/yaml/EX101.yaml
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813
