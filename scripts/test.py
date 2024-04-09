from itertools import chain
import json
import numpy as np

from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    AutoModelForTokenClassification, 
    DataCollatorForTokenClassification, 
)

data = json.load(open("/kaggle/input/pii-detection-removal-from-educational-data/train.json"))

# downsampling of negative examples
p = [] # positive samples (contain relevant labels)
n = [] # negative samples (presumably contain entities that are possibly wrongly classified as entity)
for d in data:
    if any(np.array(d["labels"]) != "O"): p.append(d)
    else: n.append(d)

data = p + n[:len(n)//3]
print("original datapoints: ", len(data))

all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
label2id = {l: i for i,l in enumerate(all_labels)}
id2label = {v:k for k,v in label2id.items()}

TRAINING_MODEL_PATH = "/kaggle/output/EX008/checkpoint-1745"

tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)

model = AutoModelForTokenClassification.from_pretrained(
    TRAINING_MODEL_PATH,
    num_labels=len(all_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
