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

# %% Evaluate during training --> do we need to do this??

# %% TRAIN
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
checkpoint = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, force_download=True)

# %%
training_args = Seq2SeqTrainingArguments(
    output_dir="trained_model", 
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True, #change to bf16=True for XPU
    push_to_hub=False,
)

# %%
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics,
)

# %%
trainer.train()


# %% Test translating a sentence
text = "translate Quechua to Spanish: Jesusqa Isaiaspa nisqantam kay Pachapi Diospa munayninta ruraspan allinta cumplirqa."
# Reference Translation: "Jesús cumplió de forma sorprendente esta profecía durante su ministerio."
from transformers import pipeline

translator = pipeline("translation_qu_to_es", model="trained_model")
translator(text)