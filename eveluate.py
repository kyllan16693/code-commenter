import os
import re
import torch
import numpy as np
import nltk
from nltk import word_tokenize
from langdetect import detect
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq
)
import evaluate

# Download NLTK data
nltk.download('punkt')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer and model
model_path = './final_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

# Report model summary
print("Model Summary:")
print(model)
print("\n")

# Report total and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# Report model size
def get_model_size(model_dir):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

model_size = get_model_size(model_path)
print(f"Model size (bytes): {model_size}")
print(f"Model size (MB): {model_size / (1024*1024):.2f} MB")
print("\n")

# Load the test dataset
dataset = load_dataset("code_search_net", "python", streaming=True)
test_dataset = dataset['test']

# Define preprocessing functions
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def clean_docstring(docstring):
    docstring = re.sub(r"@\w+", "", docstring)
    docstring = re.sub(r"<.*?>", "", docstring)
    docstring = re.sub(r"\s+", " ", docstring)
    docstring = docstring.strip()
    return docstring

def filter_and_preprocess(dataset):
    # Filter out non-English docstrings
    dataset = dataset.filter(lambda x: is_english(x['func_documentation_string']))
    # Clean docstrings
    dataset = dataset.map(lambda x: {'func_documentation_string': clean_docstring(x['func_documentation_string'])})
    return dataset

# Filter and preprocess test dataset
test_dataset = filter_and_preprocess(test_dataset)

# Define the tokenize function
def tokenize_function(example):
    # Tokenize inputs
    model_inputs = tokenizer(
        example['func_code_string'],
        max_length=512,
        truncation=True,
        padding='max_length'
    )
    # Tokenize labels using 'text_target' argument
    model_inputs['labels'] = tokenizer(
        text_target=example['func_documentation_string'],
        max_length=128,
        truncation=True,
        padding='max_length'
    )['input_ids']
    return model_inputs

# Columns to remove
columns_to_remove = [
    'repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string',
    'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string',
    'func_documentation_tokens', 'split_name', 'func_code_url'
]

# Tokenize the test dataset
tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)

# Create DataCollator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Create DataLoader for the test dataset
from torch.utils.data import DataLoader

test_dataloader = DataLoader(
    tokenized_test,
    batch_size=8,
    collate_fn=data_collator
)

# Set model to evaluation mode
model.eval()

# Initialize lists to collect predictions and references
all_predictions = []
all_references = []

# Generate predictions
print("Generating predictions on the test dataset...")
for batch in test_dataloader:
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=4
        )

    # Decode the outputs and labels
    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Replace -100 in labels as we can't decode them
    labels = labels.cpu().numpy()
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Append to lists
    all_predictions.extend(decoded_preds)
    all_references.extend(decoded_labels)

# Load evaluation metrics
bleu_metric = evaluate.load('bleu')
rouge_metric = evaluate.load('rouge')

# Prepare references in the format expected by the metrics
all_references_list = [[ref] for ref in all_references]

# Compute BLEU score
bleu_result = bleu_metric.compute(predictions=all_predictions, references=all_references_list)
print(f"BLEU Score: {bleu_result['bleu']:.4f}")

# Compute ROUGE score
rouge_result = rouge_metric.compute(predictions=all_predictions, references=all_references)
print(f"ROUGE Scores:")
for key in rouge_result.keys():
    print(f"  {key}: {rouge_result[key]:.4f}")

# Compute exact match accuracy
exact_matches = [int(pred.strip() == ref.strip()) for pred, ref in zip(all_predictions, all_references)]
accuracy = sum(exact_matches) / len(exact_matches)
print(f"Exact Match Accuracy: {accuracy:.4f}")
