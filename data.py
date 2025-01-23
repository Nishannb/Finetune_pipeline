from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")

dataset = load_dataset("samsum")


tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])

input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
print(input_lenghts)
# take 85 percentile of max length for better utilization
max_source_length = int(np.percentile(input_lenghts, 85))
print(f"Max source length: {max_source_length}")
print(f"max: {np.max(input_lenghts)}" )

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# take 90 percentile of max length for better utilization
max_target_length = int(np.percentile(target_lenghts, 90))
print(f"Max target length: {max_target_length}")
print(f"min: {np.min(input_lenghts)}" )

# print(f"Train dataset size: {len(dataset['train'])}")
# print(f"Test dataset size: {len(dataset['test'])}")
