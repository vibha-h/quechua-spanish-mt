# %% load dataset
from datasets import load_dataset
ds = load_dataset("somosnlp-hackathon-2022/spanish-to-quechua")


# %% Split the dataset into a train and test set with the train_test_split method:
ds = ds["train"].train_test_split(test_size=0.2)

# %%
ds["train"][10] 

# %% load a T5 tokenizer to process quechua-spanish pairs
from transformers import AutoTokenizer
checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint) 

# %%
source_lang = "qu"
target_lang = "es"
prefix = "translate Quechua to Spanish: "

def preprocess_function(examples):
    inputs = [prefix + text for text in examples[source_lang]]
    targets = examples[target_lang]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    print("Sample input:", inputs[0] if inputs else "no inputs")
    print("Sample target:", targets[0] if targets else "no targets")
    return model_inputs

# %%
tokenized_ds = ds.map(preprocess_function, batched=True)

# %%
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# %% Evaluate during training
import evaluate
metric = evaluate.load("sacrebleu")

# %% function that passes your predictions and labels to compute to calculate the SacreBLEU score
