#/bin/bash

# kaggle datasets create -p /home/ubuntu/my_kaggle_docker/output/EX006 --dir-mode zip > /home/ubuntu/my_kaggle_docker/output/EX006/log2.txt
kaggle datasets create -p /home/ubuntu/my_kaggle_docker/output/EX007 --dir-mode zip > /home/ubuntu/my_kaggle_docker/output/EX007/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813	
