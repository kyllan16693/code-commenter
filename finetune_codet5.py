# Import required libraries
from datasets import load_dataset, load_from_disk

from transformers import (
    RobertaTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset, DataLoader
# Load the CodeSearchNet dataset
print("Loading dataset...")
dataset = load_dataset("code_search_net", "python")

# Load the tokenizer and model
print("Loading tokenizer and model...")
tokenizer = RobertaTokenizer.from_pretrained(
    "Salesforce/codet5-base", trust_remote_code=True
)
model = T5ForConditionalGeneration.from_pretrained(
    "Salesforce/codet5-base-multi-sum",
    trust_remote_code=True
)


def preprocess_function(examples):
    inputs = tokenizer(
        examples["func_code_string"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    labels = tokenizer(
        [" ".join(doc) for doc in examples["func_documentation_tokens"]],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    labels["input_ids"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in labels["input_ids"]
    ]
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"],
    }

def filter_data(row):
    """
    Keep rows where the length of func_documentation_tokens is less than 11.
    """
    return len(row["func_documentation_tokens"]) < 11

print("Preprocessing dataset...")
tokenized_train = dataset["train"].filter(filter_data).map(
    preprocess_function, batched=True, remove_columns=dataset["train"].column_names
)
tokenized_eval = dataset["validation"].filter(filter_data).map(
    preprocess_function, batched=True, remove_columns=dataset["validation"].column_names
)
tokenized_train.save_to_disk("tokenized_train")
tokenized_eval.save_to_disk("tokenized_eval")

train_dataset = load_from_disk("tokenized_train")
eval_dataset = load_from_disk("tokenized_eval")



# Dynamic padding
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="codet5_model_out",
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    learning_rate=5e-5,
    evaluation_strategy="steps",
    eval_steps=1000,
    warmup_steps=500,
    weight_decay=0.01,
    load_best_model_at_end=True,
    remove_unused_columns=True,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the model and tokenizer
model.save_pretrained("codet5_model_out")
tokenizer.save_pretrained("codet5_model_out")
