# Text classification
''' 
Text classification is a common NLP task that assigns a label or class to text.

Metrics:
1. Accuracy
2.F1 score
'''
from dotenv import load_dotenv
load_dotenv()

import os
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"),add_to_git_credential=True)

from datasets import load_dataset

imdb=load_dataset("imdb")
#print(imdb["test"][0])

from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"],truncation=True)

tokenized_imdb=imdb.map(preprocess_function,batched=True)

from transformers import DataCollatorWithPadding

data_collator=DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate
accuracy=evaluate.load("accuracy")

import numpy as np

def compute_metrics(eval_pred):
    predictions,labels=eval_pred
    predictions=np.argmax(predictions,axis=1)
    return accuracy.compute(predictions=predictions,references=labels)

id2label={0:"NEGATIVE",1:"POSITIVE"}
label2id={"NEGATIVE":0,"POSITIVE":1}

from transformers import TrainingArguments,AutoModelForSequenceClassification,Trainer

model=AutoModelForSequenceClassification.from_pretrained('distilbert/distilbert-base-uncased',num_labels=2,id2label=id2label,label2id=label2id)

training_args=TrainingArguments(
    output_dir='text_classifier',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy='epoch',
    load_best_model_at_end=True,
    push_to_hub=True,
)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.push_to_hub()
