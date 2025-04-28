# %% load dataset
from datasets import load_dataset
ds = load_dataset("somosnlp-hackathon-2022/spanish-to-quechua")

# %% Split the dataset into a train and test set with the train_test_split method:
ds = ds["train"].select(range(10000)).train_test_split(test_size=0.2)

# %%
ds["train"][10] 

# %% load a T5 tokenizer to process quechua-spanish pairs
from transformers import AutoTokenizer
model_path = "trained_model/checkpoint-2500"
checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_path)

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
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# %%
training_args = Seq2SeqTrainingArguments(
    output_dir="trained_model", 
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
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
#text = "translate Quechua to Spanish: Jesusqa Isaiaspa nisqantam kay Pachapi Diospa munayninta ruraspan allinta cumplirqa."
# Reference Translation: "Jesús cumplió de forma sorprendente esta profecía durante su ministerio."
from transformers import pipeline

#translator = pipeline("translation_qu_to_es", model="trained_model/checkpoint-100")
translator = pipeline("translation_qu_to_es", model=model, tokenizer=tokenizer)
#result = translator(text, max_length=500) 
#print("Translation:", result[0]['translation_text'])

#list of phrases (Quechua, Reference Spanish)
phrases = [
    ("ñuqa aycha-ta-m miku-ni", "yo como carne"),
    ("Pitaq kanki?", "¿quién eres?"),
    ("Yachay wasinchikpi", "En nuestra casa de estudios"),
    ("Yachachiq yachachisqakunapas yachay wasi ukupi kachkanku.", "El profesor y sus alumnos están en el aula."),
    ("Allinllam, yachachiqniy, qamrí?", "Estamos bien, mi profesor. ¿Y tú?"),
    ("Ñuqataq San Isidropi tiyachkani.", "Yo, por mi parte, vivo en San Isidro."),
    ("Ñuqaqa mamaypaq yanuqmi kani, qamrí, yaw Ricardo?", "Yo suelo cocinar para mi mamá, ¿y tú, oye, Ricardo?"),
    ("Arí, ñuqaqa futbolpi pukllaqmi kani.", "Sí, yo suelo jugar fútbol."),
    ("Haykaptaq qawaytarí tukunki?", "¿Cuándo vas a terminar la revista?"),
    ("Imaynataq kachkan?", "¿Cómo está?")
]

translated_phrases = []

for idx, (quechua_text, reference_spanish) in enumerate(phrases, start=1):
    model_input = f"translate Quechua to Spanish: {quechua_text}"
    #print(model_input)
    machine_translation = translator(model_input, max_length=500)[0]['translation_text']
    translated_phrases.append((quechua_text, machine_translation, reference_spanish))

    print(f"Phrase {idx}:")
    print(f"Quechua: {quechua_text}")
    print(f"Machine Translation: {machine_translation}")
    print(f"Reference Spanish: {reference_spanish}")
    print()


