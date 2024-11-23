import re
import nltk
nltk.download('punkt')
from langdetect import detect
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import google.protobuf
import transformers
import accelerate
import torch
import numpy as np


import transformers
import accelerate
import torch

print("Transformers version:", transformers.__version__)
print("Accelerate version:", accelerate.__version__)
print("Torch version:", torch.__version__)





# Load the dataset
dataset = load_dataset("code_search_net", "python")
train_dataset = dataset['train']
valid_dataset = dataset['validation']
test_dataset = dataset['test']

# View sample data
sample = train_dataset[0]
print("Sample keys:", sample.keys())
print("Code:\n", sample['func_code_string'])
print("\nDocstring:\n", sample['func_documentation_string'])


# Define functions
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

# Filter and preprocess datasets
train_dataset = filter_and_preprocess(train_dataset)
valid_dataset = filter_and_preprocess(valid_dataset)
test_dataset = filter_and_preprocess(test_dataset)

# Sample 10% of the dataset
#train_dataset = train_dataset.shuffle(seed=42).select(range(int(0.1 * len(train_dataset))))
#valid_dataset = valid_dataset.shuffle(seed=42).select(range(int(0.1 * len(valid_dataset))))
#test_dataset = test_dataset.shuffle(seed=42).select(range(int(0.1 * len(test_dataset))))


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('SEBIS/code_trans_t5_base_code_documentation_generation_java')
model = AutoModelForSeq2SeqLM.from_pretrained('SEBIS/code_trans_t5_base_code_documentation_generation_java')

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Tokenize datasets
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

columns_to_remove = [
    'repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string',
    'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string',
    'func_documentation_tokens', 'split_name', 'func_code_url'
]

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
tokenized_valid = valid_dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    predict_with_generate=True,
    generation_max_length=128, # Or max_new_tokens=128 ?
    #num_beams=4,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# Load BLEU metric
metric = load_metric('bleu')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Convert predictions and labels to numpy arrays if they are tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
        
    # Check for NaNs and Infs
    if np.isnan(predictions).any() or np.isinf(predictions).any():
        print("Predictions contain NaNs or Infs")
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Ensure predictions are integers
    predictions = predictions.astype(np.int64)
    labels = labels.astype(np.int64)

    # Check for invalid token IDs
    max_token_id = tokenizer.vocab_size
    if predictions.max() >= max_token_id or predictions.min() < 0:
        print("Invalid token IDs in predictions")
        # Clip the values to valid token IDs
        predictions = np.clip(predictions, 0, max_token_id - 1)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip leading and trailing spaces
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Wrap references in a list for BLEU score
    decoded_labels = [[label] for label in decoded_labels]

    # Compute BLEU score
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    # Return the result
    return {"bleu": result["bleu"]}

# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# Train model
trainer.train()

# Evaluate model
eval_results = trainer.evaluate(tokenized_test)
print(eval_results)

# Generate code comments
def generate_comment(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", max_length=512, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.generate(
        inputs['input_ids'],
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    comment = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return comment

# Example usage
code_example = test_dataset[0]['func_code_string']
print("Code:\n", code_example)
print("\nGenerated Comment:\n", generate_comment(code_example))

# save model
trainer.save_model('./final_model')
tokenizer.save_pretrained('./final_model')
