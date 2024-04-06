#/bin/bash

python /kaggle/scripts/train_4folds_freeze.py --config /kaggle/scripts/yaml/deberta3large_0966_bf16_4folds0_freeze_gc.yaml > /kaggle/output/deberta3large_gc_fold0_freeze6/log.txt
python /kaggle/scripts/train_4folds_freeze_AWP2.py --config /kaggle/scripts/yaml/deberta3large_0966_bf16_4folds0_freeze_AWP2_wur1_gc.yaml > /kaggle/output/deberta3large_wur1_gc_fold0_freeze6_AWP2/log.txt
kaggle datasets create -p /kaggle/output/deberta3large_gc_fold0_freeze6 --dir-mode zip > /kaggle/output/deberta3large_gc_fold0_freeze6/log2.txt
kaggle datasets create -p /kaggle/output/deberta3large_wur1_gc_fold0_freeze6_AWP2 --dir-mode zip > /kaggle/output/deberta3large_wur1_gc_fold0_freeze6_AWP2/log2.txt
aws ec2 stop-instances --instance-ids i-0449cb0b94ddaa813	
