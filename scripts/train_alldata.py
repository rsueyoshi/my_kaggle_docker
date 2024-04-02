from dotenv import load_dotenv
import os

load_dotenv("/kaggle/.env")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


import json
import argparse
from itertools import chain
from functools import partial
import random
import sys
from types import SimpleNamespace
from typing import Optional, Tuple, Union
import yaml

from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    AutoModelForTokenClassification, 
    DataCollatorForTokenClassification, 
)
from datasets import Dataset
import numpy as np
import torch

import wandb
from seqeval.metrics import recall_score, precision_score

def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)
cfg = yaml.safe_load(open(parser_args.config).read())

for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)

print(cfg)


TRAINING_MODEL_PATH = cfg.architecture.backbone
TRAINING_MAX_LENGTH = cfg.tokenizer.max_length
STRIDE = cfg.tokenizer.stride
if cfg.architecture.freeze_layers:
    OUTPUT_DIR = f"{cfg.output.dir}_freeze{cfg.architecture.freeze_layers}"
    name = f"{cfg.architecture.name}_freeze{cfg.architecture.freeze_layers}"
else:
    OUTPUT_DIR = cfg.output.dir
    name = cfg.architecture.name

BATCH_SIZE = cfg.training.batch_size
ACC_STEPS = cfg.training.grad_accumulation
EPOCHS = cfg.training.epochs
LR = cfg.training.learning_rate

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(cfg.environment.seed)


wandb.login()

run = wandb.init(
    project="kaggle_pii",
    name=name,
    config=cfg
)

data = json.load(open(cfg.dataset.json_filepath))

# downsampling of negative examples
p = [] # positive samples (contain relevant labels)
n = [] # negative samples (presumably contain entities that are possibly wrongly classified as entity)
for d in data:
    if any(np.array(d["labels"]) != "O"): p.append(d)
    else: n.append(d)
print("original datapoints: ", len(data))

data = p + n[:len(n)//3]

print("filtered datapoints: ", len(data))

for i, ex_filepath in enumerate(cfg.dataset.extra_filepath):
    ex_data = json.load(open(ex_filepath))
    print(f"external datapoints {i}: {len(ex_data)}")
    data = data + ex_data

print("combined: ", len(data))

all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
label2id = {l: i for i,l in enumerate(all_labels)}
id2label = {v:k for k,v in label2id.items()}

print(id2label)


def tokenize_train(example, tokenizer, label2id):

    # rebuild text from tokens
    text = []
    labels = []

    for idx, (t, l, ws) in enumerate(zip(
        example["tokens"], example["provided_labels"], example["trailing_whitespace"]
    )):
        text.append(t)
        labels.extend([l] * len(t))

        if ws:
            text.append(" ")
            labels.append("O")

    # actual tokenization
    tokenized = tokenizer(
        "".join(text),
        return_offsets_mapping=True,
        max_length=TRAINING_MAX_LENGTH, 
        stride=STRIDE,
        truncation=True, 
        return_overflowing_tokens=True,
    )

    labels = np.array(labels)

    text = "".join(text)
    token_labels = []
    
    for offsets in tokenized.offset_mapping:
        tmp_labels = []
        
        for start_idx, end_idx in offsets:        
            # CLS token
            if start_idx == 0 and end_idx == 0:
                tmp_labels.append(-100)
                continue

            # case when token starts with whitespace
            if text[start_idx].isspace():
                start_idx += 1

            tmp_labels.append(label2id[labels[start_idx]])
        token_labels.append(tmp_labels)

    tokenized.pop("overflow_to_sample_mapping")
    tokenized.pop("offset_mapping")
    return {
        **tokenized, 
        "labels": token_labels, 
    }


tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)

if cfg.debug:
    n = 64
    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data[:n]],
        "document": [str(x["document"]) for x in data[:n]],
        "tokens": [x["tokens"] for x in data[:n]],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data[:n]],
        "provided_labels": [x["labels"] for x in data[:n]],
    })
else:
    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [str(x["document"]) for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        "provided_labels": [x["labels"] for x in data],
    })

ds = ds.map(
    tokenize_train, 
    fn_kwargs={"tokenizer": tokenizer, "label2id": label2id}, 
    remove_columns=ds.column_names,
    num_proc=os.cpu_count()
)


train_dict = None
for d in ds:
    if train_dict is None:
        train_dict = d
    else:
        for k, v in d.items():
            train_dict[k] += d[k]

ds = Dataset.from_dict(train_dict)


def compute_metrics(p, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f1_score = (1 + 5*5) * recall * precision / (5*5*precision + recall)
    
    results = {
        'recall': recall,
        'precision': precision,
        'f1': f1_score
    }
    return results

if cfg.model_class == "DebertaV2ForTokenClassification":
    model = AutoModelForTokenClassification.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

elif cfg.model_class == "DebertaV2CnnForTokenClassification":
    from models.debertav2cnn import DebertaV2CnnForTokenClassification
    model = DebertaV2CnnForTokenClassification.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

elif cfg.model_class == "DebertaV2LstmForTokenClassification":
    from models.debertav2cnn import DebertaV2LstmForTokenClassification
    model = DebertaV2LstmForTokenClassification.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

if cfg.architecture.freeze_layers:
    for param in model.deberta.embeddings.parameters():
        param.requires_grad = False if cfg.architecture.freeze_embedding else True
    for layer in model.deberta.encoder.layer[:cfg.architecture.freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False

collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)


# I actually chose to not use any validation set. This is only for the model I use for submission.
if cfg.training.evaluation_strategy == "no":
    do_eval = False
else:
    do_eval = True

args = TrainingArguments(
    output_dir=OUTPUT_DIR, 
    bf16=cfg.training.bf16,
    fp16=cfg.training.fp16,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACC_STEPS,
    report_to="wandb",
    evaluation_strategy=cfg.training.evaluation_strategy,
    do_eval=do_eval,
    logging_steps=20,
    lr_scheduler_type=cfg.training.schedule,
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=ds,
    data_collator=collator, 
    tokenizer=tokenizer,
    compute_metrics=partial(compute_metrics, all_labels=all_labels),
)

trainer.train()


wandb.finish()
