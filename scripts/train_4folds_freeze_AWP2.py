from dotenv import load_dotenv
import os

load_dotenv("/kaggle/.env")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


from collections import defaultdict
import copy
import glob
import json
import argparse
from itertools import chain
from functools import partial
import random
import re
import sys
from types import SimpleNamespace
from typing import Optional, Tuple, Union
import yaml

from transformers import (
    AutoTokenizer, 
    # Trainer, 
    TrainingArguments,
    AutoModelForTokenClassification, 
    DataCollatorForTokenClassification, 
    EvalPrediction,
)
from datasets import Dataset, DatasetDict, concatenate_datasets
import numpy as np
import pandas as pd
import torch
from spacy.lang.en import English

import wandb

from utils.trainer_utils import Trainer_Awp as Trainer
from utils.collator import Collator

parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)
cfg = yaml.safe_load(open(parser_args.config).read())
cfg_tmp = copy.deepcopy(cfg)

for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)

print(cfg)

TRAINING_MODEL_PATH = cfg.architecture.backbone
TRAINING_MAX_LENGTH = cfg.tokenizer.max_length
STRIDE = cfg.tokenizer.stride
FOLD = cfg.fold.fold
OUTPUT_DIR = f"/kaggle/output/{cfg.architecture.name}"

BATCH_SIZE = cfg.training.batch_size
EVAL_BATCH_SIZE = cfg.training.eval_batch_size
ACC_STEPS = cfg.training.grad_accumulation
EVAL_STEPS = cfg.training.eval_steps
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

checkpoints = glob.glob(f"{OUTPUT_DIR}/checkpoint-*")
if len(checkpoints) > 0:
    resume=True
else:
    resume=False

run = wandb.init(
    project="kaggle_pii",
    name=f"{cfg.architecture.name}",
    config=cfg,
    resume=resume,
)

original_data = json.load(open(cfg.dataset.json_filepath))

ex_data_list = []
for i, ex_filepath in enumerate(cfg.dataset.extra_filepath):
    ex_data = json.load(open(ex_filepath))
    print(f"external datapoints {i}: {len(ex_data)}")
    ex_data_list.append(ex_data)

all_labels = sorted(list(set(chain(*[x["labels"] for x in original_data]))))
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


def tokenize_valid(example, tokenizer, label2id):

    # rebuild text from tokens
    text = []
    labels = []
    token_map = []


    for idx, (t, l, ws) in enumerate(zip(
        example["tokens"], example["provided_labels"], example["trailing_whitespace"]
    )):
        text.append(t)
        labels.extend([l] * len(t))
        token_map.extend([idx]*len(t))

        if ws:
            text.append(" ")
            labels.append("O")
            token_map.append(-1)

    # actual tokenization
    tokenized = tokenizer("".join(text), return_offsets_mapping=True,
                        truncation=False)

    labels = np.array(labels)

    text = "".join(text)
    token_labels = []
    
    for start_idx, end_idx in tokenized.offset_mapping:
        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(-100)
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        token_labels.append(label2id[labels[start_idx]])
    
    length = len(tokenized.input_ids)

    return {
        **tokenized, 
        "labels": token_labels, 
        "length": length,
        "token_map": token_map,
    }


tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)

ds = DatasetDict()

ex_data_names = [f"extra_{i}" for i in range(len(ex_data_list))]

if cfg.debug:
    n = 128
    for key, data in zip(["original"], [original_data]):
        ds[key] = Dataset.from_dict({
            "full_text": [x["full_text"] for x in data[:n]],
            "document": [str(x["document"]) for x in data[:n]],
            "tokens": [x["tokens"] for x in data[:n]],
            "trailing_whitespace": [x["trailing_whitespace"] for x in data[:n]],
            "provided_labels": [x["labels"] for x in data[:n]],
        })
else:
    for key, data in zip(["original", *ex_data_names], [original_data, *ex_data_list]):
        ds[key] = Dataset.from_dict({
            "full_text": [x["full_text"] for x in data],
            "document": [str(x["document"]) for x in data],
            "tokens": [x["tokens"] for x in data],
            "trailing_whitespace": [x["trailing_whitespace"] for x in data],
            "provided_labels": [x["labels"] for x in data],
        })

N_SPLITS = cfg.fold.num_folds
folds = [
    (
        np.array([i for i, d in enumerate(ds["original"]["document"]) if int(d) % N_SPLITS != s]),
        np.array([i for i, d in enumerate(ds["original"]["document"]) if int(d) % N_SPLITS == s])
    )
    for s in range(N_SPLITS)
]

negative_idxs = [i for i, labels in enumerate(ds["original"]["provided_labels"]) if not any(np.array(labels) != "O")]
exclude_indices = negative_idxs[int(len(negative_idxs) * cfg.dataset.negative_ratio):]

train_idx, eval_idx = folds[FOLD]

original_ds = ds["original"].select([i for i in train_idx if i not in exclude_indices])
if cfg.debug:
    train_ds = original_ds
else:
    train_ds = concatenate_datasets([original_ds, *[ds[name] for name in ex_data_names]])
train_ds = train_ds.map(
    tokenize_train, 
    fn_kwargs={"tokenizer": tokenizer, "label2id": label2id}, 
    remove_columns=train_ds.column_names,
    num_proc=os.cpu_count()
)

train_dict = None
for d in train_ds:
    if train_dict is None:
        train_dict = d
    else:
        for k, v in d.items():
            train_dict[k] += d[k]

train_ds = Dataset.from_dict(train_dict)

eval_ds = ds["original"].select(eval_idx)
eval_ds = eval_ds.map(
    tokenize_valid, 
    fn_kwargs={"tokenizer": tokenizer, "label2id": label2id}, 
    num_proc=os.cpu_count()
)

def find_span(target: list[str], document: list[str]) -> list[list[int]]:
    idx = 0
    spans = []
    span = []

    for i, token in enumerate(document):
        if token != target[idx]:
            idx = 0
            span = []
            continue
        span.append(i)
        idx += 1
        if idx == len(target):
            spans.append(span)
            span = []
            idx = 0
            continue

    return spans


class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):  # in-place add
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def f5(self) -> float:
        beta = 5
        p = self.precision
        r = self.recall

        fbeta = (1+(beta**2))*p*r / ((beta**2)*p + r + 1e-100)
        return fbeta

    def to_dict(self) -> dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}


class MetricsComputer:
    nlp = English()

    def __init__(self, eval_ds: Dataset, label2id: dict, conf_thresh: float = 0.9) -> None:
        self.ds = eval_ds.remove_columns("labels").rename_columns({"provided_labels": "labels"})
        self.gt_df = self.create_gt_df(self.ds)
        self.label2id = label2id
        self.confth = conf_thresh
        self._search_gt()

    def __call__(self, eval_preds: EvalPrediction) -> dict:
        pred_df = self.create_pred_df(eval_preds.predictions)
        return self.compute_metrics_from_df(self.gt_df, pred_df)

    def _search_gt(self) -> None:
        email_regex = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
        phone_num_regex = re.compile(r"(\(\d{3}\)\d{3}\-\d{4}\w*|\d{3}\.\d{3}\.\d{4})\s")
        self.emails = []
        self.phone_nums = []

        for _data in self.ds:
            # email
            for token_idx, token in enumerate(_data["tokens"]):
                if re.fullmatch(email_regex, token) is not None:
                    self.emails.append(
                        {"document": _data["document"], "token": token_idx, "label": "B-EMAIL", "token_str": token}
                    )
            # phone number
            matches = phone_num_regex.findall(_data["full_text"])
            if not matches:
                continue
            for match in matches:
                target = [t.text for t in self.nlp.tokenizer(match)]
                matched_spans = find_span(target, _data["tokens"])
            for matched_span in matched_spans:
                for intermediate, token_idx in enumerate(matched_span):
                    prefix = "I" if intermediate else "B"
                    self.phone_nums.append(
                        {"document": _data["document"], "token": token_idx, "label": f"{prefix}-PHONE_NUM", "token_str": _data["tokens"][token_idx]}
                    )

    @staticmethod
    def create_gt_df(ds: Dataset):
        gt = []
        for row in ds:
            for token_idx, (token, label) in enumerate(zip(row["tokens"], row["labels"])):
                if label == "O":
                    continue
                gt.append(
                    {"document": row["document"], "token": token_idx, "label": label, "token_str": token}
                )
        gt_df = pd.DataFrame(gt)
        gt_df["row_id"] = gt_df.index

        return gt_df

    def create_pred_df(self, logits: np.ndarray) -> pd.DataFrame:
        """
        Note:
            Thresholing is doen on logits instead of softmax, which could find better models on LB.
        """
        prediction = logits
        o_index = self.label2id["O"]
        preds = prediction.argmax(-1)
        preds_without_o = prediction.copy()
        preds_without_o[:,:,o_index] = 0
        preds_without_o = preds_without_o.argmax(-1)
        o_preds = prediction[:,:,o_index]
        preds_final = np.where(o_preds < self.confth, preds_without_o , preds)

        pairs = set()
        processed = []

        # Iterate over document
        for p_doc, token_map, offsets, tokens, doc in zip(
            preds_final, self.ds["token_map"], self.ds["offset_mapping"], self.ds["tokens"], self.ds["document"]
        ):
            # Iterate over sequence
            for p_token, (start_idx, end_idx) in zip(p_doc, offsets):
                label_pred = id2label[p_token]

                if start_idx + end_idx == 0:
                    # [CLS] token i.e. BOS
                    continue

                if token_map[start_idx] == -1:
                    start_idx += 1

                # ignore "\n\n"
                while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                    start_idx += 1

                if start_idx >= len(token_map):
                    break

                token_id = token_map[start_idx]
                pair = (doc, token_id)

                # ignore "O", preds, phone number and  email
                if label_pred in ("O", "B-EMAIL", "B-PHONE_NUM", "I-PHONE_NUM") or token_id == -1:
                    continue

                if pair in pairs:
                    continue

                processed.append(
                    {"document": doc, "token": token_id, "label": label_pred, "token_str": tokens[token_id]}
                )
                pairs.add(pair)

        pred_df = pd.DataFrame(processed + self.emails + self.phone_nums)
        pred_df["row_id"] = list(range(len(pred_df)))

        return pred_df

    def compute_metrics_from_df(self, gt_df, pred_df):
        """
        Compute the LB metric (lb) and other auxiliary metrics
        """

        references = {(row.document, row.token, row.label) for row in gt_df.itertuples()}
        predictions = {(row.document, row.token, row.label) for row in pred_df.itertuples()}

        score_per_type = defaultdict(PRFScore)
        references = set(references)

        for ex in predictions:
            pred_type = ex[-1] # (document, token, label)
            if pred_type != 'O':
                pred_type = pred_type[2:] # avoid B- and I- prefix

            if pred_type not in score_per_type:
                score_per_type[pred_type] = PRFScore()

            if ex in references:
                score_per_type[pred_type].tp += 1
                references.remove(ex)
            else:
                score_per_type[pred_type].fp += 1

        for doc, tok, ref_type in references:
            if ref_type != 'O':
                ref_type = ref_type[2:] # avoid B- and I- prefix

            if ref_type not in score_per_type:
                score_per_type[ref_type] = PRFScore()
            score_per_type[ref_type].fn += 1

        totals = PRFScore()

        for prf in score_per_type.values():
            totals += prf

        return {
            "precision": totals.precision,
            "recall": totals.recall,
            "f5": totals.f5,
            **{
                f"{v_k}-{k}": v_v
                for k in set([l[2:] for l in self.label2id.keys() if l!= 'O'])
                for v_k, v_v in score_per_type[k].to_dict().items()
            },
        }

if cfg.model_class == "DebertaV2ForTokenClassification":
    model = AutoModelForTokenClassification.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

elif cfg.model_class == "DebertaDo5ForTokenClassification":
    from models.debertado5 import DebertaDo5ForTokenClassification
    model = DebertaDo5ForTokenClassification.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

elif cfg.model_class == "DebertaV2Do5ForTokenClassification":
    from models.debertav2do5 import DebertaV2Do5ForTokenClassification
    model = DebertaV2Do5ForTokenClassification.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

elif cfg.model_class == "DebertaV2Do5ForTokenClassification_6hiddenstates":
    from models.debertav2do5_6hiddenstates import DebertaV2Do5ForTokenClassification_6hiddenstates
    model = DebertaV2Do5ForTokenClassification_6hiddenstates.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

elif cfg.model_class == "DebertaV2Do5ForTokenClassification_12hiddenstates":
    from models.debertav2do5_12hiddenstates import DebertaV2Do5ForTokenClassification_12hiddenstates
    model = DebertaV2Do5ForTokenClassification_12hiddenstates.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

elif cfg.model_class == "DebertaCnnForTokenClassification":
    from models.debertacnn import DebertaCnnForTokenClassification
    model = DebertaCnnForTokenClassification.from_pretrained(
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


elif cfg.model_class == "DebertaLstmForTokenClassification":
    from models.debertalstm import DebertaLstmForTokenClassification
    model = DebertaLstmForTokenClassification.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

elif cfg.model_class == "DebertaV2LstmForTokenClassification":
    from models.debertav2lstm import DebertaV2LstmForTokenClassification
    model = DebertaV2LstmForTokenClassification.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

elif cfg.model_class == "RobertaDo5ForTokenClassification":
    from models.robertado5 import RobertaDo5ForTokenClassification
    model = RobertaDo5ForTokenClassification.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

elif cfg.model_class == "LongformerDo5ForTokenClassification":
    from models.longformerdo5 import LongformerDo5ForTokenClassification
    model = LongformerDo5ForTokenClassification.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

if "Deberta" in cfg.model_class:
    for param in model.deberta.embeddings.parameters():
        param.requires_grad = False if cfg.architecture.freeze_embedding else True
    for layer in model.deberta.encoder.layer[:cfg.architecture.freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False
elif "Roberta" in cfg.model_class:
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False if cfg.architecture.freeze_embedding else True
    for layer in model.roberta.encoder.layer[:cfg.architecture.freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False
elif "Longformer" in cfg.model_class:
    for param in model.longformer.embeddings.parameters():
        param.requires_grad = False if cfg.architecture.freeze_embedding else True
    for layer in model.longformer.encoder.layer[:cfg.architecture.freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False

collator = Collator(tokenizer)

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
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=ACC_STEPS,
    report_to="wandb",
    evaluation_strategy=cfg.training.evaluation_strategy,
    do_eval=do_eval,
    load_best_model_at_end=do_eval,
    eval_steps=EVAL_STEPS,
    eval_delay=EVAL_STEPS*6,
    save_strategy=cfg.training.evaluation_strategy,
    save_steps=EVAL_STEPS,
    save_total_limit=2,
    greater_is_better=True,
    overwrite_output_dir=True,
    logging_steps=20,
    lr_scheduler_type=cfg.training.schedule,
    metric_for_best_model=cfg.training.metric_for_best_model,
    warmup_ratio=0.1,
    weight_decay=0.01,
    gradient_checkpointing=cfg.training.gradient_checkpointing,
)

metrics_computer = MetricsComputer(eval_ds=eval_ds, label2id=label2id)
trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator, 
    tokenizer=tokenizer,
    compute_metrics=metrics_computer,
    adv_param=cfg.training.adv_param,
    adv_lr=cfg.training.adv_lr, 
    adv_eps=cfg.training.adv_eps, 
    awp_start=cfg.training.awp_start,
)

trainer.train(resume_from_checkpoint=resume)


# 推論結果の可視化
cols = ["ID", "ground_truth", "prediction", "f5_score", "precision", "recall"]
cols2 = ["ID"]
for _, v in id2label.items():
    cols2.append(f"Logits_{v}")
    
html_prefix = '<mark class=\"entity\" style=\"background: {color}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\"> '
html_sufix = ' <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">{label}</span></mark>'

colors = ["red", "crimson", "green", "forestgreen", "blue", "navy", "orange", "darkorange", "brown", "maroon", "deeppink", "magenta"]
color_map = {i: v for i, v in enumerate(colors)}


predictions = trainer.predict(eval_ds).predictions

if cfg.debug:
    pred_df = metrics_computer.gt_df
else:
    pred_df = metrics_computer.create_pred_df(predictions)
#%%
wb_table = wandb.Table(columns=cols)
for tokens, doc in zip(eval_ds["tokens"], eval_ds["document"]):
    doc_gt_df = metrics_computer.gt_df[metrics_computer.gt_df["document"] == doc].reset_index(drop=True)
    doc_pred_df = pred_df[pred_df["document"] == doc].reset_index(drop=True)
    gt_row_idx = 0
    pred_row_idx = 0
    text_gt = ""
    text_pred = ""
    for i, token in enumerate(tokens):
        if len(text_gt) > 0:
            text_gt += " "
        if (gt_row_idx < len(doc_gt_df)) and (i == doc_gt_df.loc[gt_row_idx, "token"]):
            label_gt_idx = label2id[doc_gt_df.loc[gt_row_idx, "label"]]
            text_gt += html_prefix.format(color=color_map[label_gt_idx])
            text_gt += token
            text_gt += html_sufix.format(label=doc_gt_df.loc[gt_row_idx, "label"])
            gt_row_idx += 1
        else:
            text_gt += token

        if len(text_pred) > 0:
            text_pred += " "
        if (pred_row_idx < len(doc_pred_df)) and (i == doc_pred_df.loc[pred_row_idx, "token"]):
            label_pred_idx = label2id[doc_pred_df.loc[pred_row_idx, "label"]]
            text_pred += html_prefix.format(color=color_map[label_pred_idx])
            text_pred += token
            text_pred += html_sufix.format(label=doc_pred_df.loc[pred_row_idx, "label"])
            pred_row_idx += 1
        else:
            text_pred += token

    metrics = metrics_computer.compute_metrics_from_df(doc_gt_df, doc_pred_df)

    wb_table.add_data(
        doc, 
        wandb.Html(text_gt), 
        wandb.Html(text_pred), 
        metrics["f5"],
        metrics["precision"],
        metrics["recall"]
    )

wandb.log({f"compare_gt_pred_fold{FOLD}": wb_table})

wandb.finish()

cfg_tmp["is_finished"] = True
