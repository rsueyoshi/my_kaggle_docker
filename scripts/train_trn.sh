#/bin/bash

# neuron_parallel_compile python3 train_4folds_freeze_trn.py --config yaml/deberta3large_0966_bf16_4folds0_freeze_trn.yaml > /home/ubuntu/my_kaggle_docker/output/deberta3large_trn_fold0_freeze6/log.txt
neuron_parallel_compile python3 train_4folds_freeze_trn.py --config yaml/deberta3large_0966_bf16_4folds0_freeze_trn.yaml
aws ec2 stop-instances --instance-ids i-0a65a75869d4457bf

