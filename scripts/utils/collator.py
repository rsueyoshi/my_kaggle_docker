import torch


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if "token_type_ids" in batch[0]:
            output["token_type_ids"] = [sample["token_type_ids"] for sample in batch]
        output["labels"] = [sample["labels"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
            if "token_type_ids" in batch[0]:
                output["token_type_ids"] = [s + (batch_max - len(s)) * [0] for s in output["token_type_ids"]]
            output["labels"] = [s + (batch_max - len(s)) * [-100] for s in output["labels"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]
            if "token_type_ids" in batch[0]:
                output["token_type_ids"] = [(batch_max - len(s)) * [0] + s for s in output["token_type_ids"]]
            output["labels"] = [(batch_max - len(s)) * [-100] + s for s in output["labels"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if "token_type_ids" in batch[0]:
            output["token_type_ids"] = torch.tensor(output["token_type_ids"], dtype=torch.long)
        output["labels"] = torch.tensor(output["labels"], dtype=torch.long)

        return output