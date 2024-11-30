import re
import pandas as pd
import torch
from tqdm import tqdm
from blue import bleuFromMaps
from datasets import load_dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from scipy.stats import ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize tqdm for pandas
tqdm.pandas()

# Load dataset
dataset = load_dataset("code_search_net", "python")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Convert to DataFrame
test_df = test_dataset.to_pandas()

# Filter test examples with docstring length <= 10 tokens
filtered_test_df = test_df[test_df["func_documentation_tokens"].map(len) < 11]
filtered_test_df_small = filtered_test_df[:4000]


# Define batch evaluation function
def batch_evaluate(test_df, model, tokenizer, batch_size=32):
    results = []
    for i in range(0, len(test_df), batch_size):
        batch = test_df.iloc[i : i + batch_size]
        contexts = [
            re.sub(r"\"\"\"(.+?)\"\"\"", "", row["func_code_string"], flags=re.DOTALL)
            for _, row in batch.iterrows()
        ]
        input_ids = tokenizer(
            contexts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).input_ids
        generated_ids = model.generate(input_ids, max_length=128)
        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for (_, row), prediction in zip(batch.iterrows(), predictions):
            true_docstring = " ".join(row["func_documentation_tokens"])
            m2 = {"1": [[true_docstring]]}
            m1 = {"1": [[prediction]]}
            bleu_score = bleuFromMaps(m1, m2)[0]
            results.append((bleu_score, prediction))
    return results


# Load BASE models and tokenizers
print("Loading the BASE model and tokenizer...")
tokenizer_pre = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
model_pre = T5ForConditionalGeneration.from_pretrained(
    "Salesforce/codet5-base-multi-sum"
)

# Load Fine-tuned models and tokenizers
model_path = "model_out"  # Path where the model and tokenizer were saved
print("Loading the fine-tuned model and tokenizer...")
tokenizer_ft = RobertaTokenizer.from_pretrained(model_path)
model_ft = T5ForConditionalGeneration.from_pretrained(model_path)


# Evaluate BASE model
print("Evaluating BASE model...")
results_pre = batch_evaluate(filtered_test_df_small.copy(), model_pre, tokenizer_pre)
filtered_test_df_small_pre = filtered_test_df_small.copy()
filtered_test_df_small_pre["bleu_score"] = [res[0] for res in results_pre]
filtered_test_df_small_pre["generated_result"] = [res[1] for res in results_pre]

# Evaluate fine-tuned model
print("Evaluating fine-tuned model...")
results_ft = batch_evaluate(filtered_test_df_small.copy(), model_ft, tokenizer_ft)
filtered_test_df_small_ft = filtered_test_df_small.copy()
filtered_test_df_small_ft["bleu_score"] = [res[0] for res in results_ft]
filtered_test_df_small_ft["generated_result"] = [res[1] for res in results_ft]

# Merge and compare results
filtered_test_df_small_pre = filtered_test_df_small_pre.reset_index()
filtered_test_df_small_ft = filtered_test_df_small_ft.reset_index()

comparison_pre = filtered_test_df_small_pre[
    [
        "index",
        "func_code_string",
        "func_documentation_tokens",
        "bleu_score",
        "generated_result",
    ]
].rename(
    columns={"bleu_score": "bleu_score_pre", "generated_result": "generated_result_pre"}
)

comparison_ft = filtered_test_df_small_ft[
    ["index", "bleu_score", "generated_result"]
].rename(
    columns={"bleu_score": "bleu_score_ft", "generated_result": "generated_result_ft"}
)

comparison_df = pd.merge(comparison_pre, comparison_ft, on="index")

# Save comparison results
comparison_df.to_csv("output/model_comparison_results.csv", index=False)
print("Comparison results saved to 'output/model_comparison_results.csv'.")

# Perform statistical test
t_stat, p_value = ttest_rel(
    filtered_test_df_small_pre["bleu_score"], filtered_test_df_small_ft["bleu_score"]
)
print(f"Paired t-test: t={t_stat}, p={p_value}")

# Create a comparison DataFrame for visualization
df_comparison = pd.DataFrame(
    {
        "BLEU Score": pd.concat(
            [
                filtered_test_df_small_pre["bleu_score"],
                filtered_test_df_small_ft["bleu_score"],
            ]
        ),
        "Model": ["CODET5-base"] * len(filtered_test_df_small_pre)
        + ["CODET5-finetuned"] * len(filtered_test_df_small_ft),
    }
)

# Plot BLEU score comparison
sns.set_context("paper")
sns.set(style="ticks")
palette = {"CODET5-base": "skyblue", "CODET5-finetuned": "orange"}
boxplot = sns.boxplot(x="Model", y="BLEU Score", data=df_comparison, palette=palette)

plt.title("BLEU Score Benchmark")
for i, model in enumerate(df_comparison["Model"].unique()):
    model_data = df_comparison[df_comparison["Model"] == model]["BLEU Score"]
    median = model_data.median()
    plt.text(
        i,
        median,
        f"{median:.2f}",
        horizontalalignment="center",
        color="black",
        weight="bold",
    )

plt.savefig("output/bleu_score_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
