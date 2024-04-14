#/bin/bash

python train_4folds_freeze.py --config /kaggle/scripts/yaml/EX016.yaml > /kaggle/output/EX016/log.txt
python train_4folds_freeze.py --config /kaggle/scripts/yaml/EX017.yaml > /kaggle/output/EX017/log.txt
python train_alldata.py --config /kaggle/scripts/yaml/EX021.yaml > /kaggle/output/EX021/log.txt
python train_alldata.py --config /kaggle/scripts/yaml/EX022.yaml > /kaggle/output/EX022/log.txt
kaggle datasets create -p /kaggle/output/EX016 --dir-mode zip > /kaggle/output/EX016/log2.txt
kaggle datasets create -p /kaggle/output/EX017 --dir-mode zip > /kaggle/output/EX017/log2.txt
kaggle datasets create -p /kaggle/output/EX021 --dir-mode zip > /kaggle/output/EX021/log2.txt
kaggle datasets create -p /kaggle/output/EX022 --dir-mode zip > /kaggle/output/EX022/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813
