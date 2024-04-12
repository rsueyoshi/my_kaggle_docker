#/bin/bash

# python train_4folds_freeze.py --config /kaggle/scripts/yaml/EX016.yaml > /kaggle/output/EX016/log.txt
# python train_4folds_freeze.py --config /kaggle/scripts/yaml/EX017.yaml > /kaggle/output/EX017/log.txt
python train_4folds_freeze.py --config /kaggle/scripts/yaml/EX018.yaml > /kaggle/output/EX018/log.txt
python train_4folds_freeze.py --config /kaggle/scripts/yaml/EX019.yaml > /kaggle/output/EX019/log.txt
python train_4folds_freeze.py --config /kaggle/scripts/yaml/EX020.yaml > /kaggle/output/EX020/log.txt
python train_4folds_freez_resume.py --config /kaggle/scripts/yaml/EX012.yaml > /kaggle/output/EX012/log.txt
# kaggle datasets create -p /kaggle/output/EX016 --dir-mode zip > /kaggle/output/EX016/log2.txt
# kaggle datasets create -p /kaggle/output/EX017 --dir-mode zip > /kaggle/output/EX017/log2.txt
kaggle datasets create -p /kaggle/output/EX018 --dir-mode zip > /kaggle/output/EX018/log2.txt
kaggle datasets create -p /kaggle/output/EX019 --dir-mode zip > /kaggle/output/EX019/log2.txt
kaggle datasets create -p /kaggle/output/EX020 --dir-mode zip > /kaggle/output/EX020/log2.txt
kaggle datasets create -p /kaggle/output/EX012 --dir-mode zip > /kaggle/output/EX012/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813
