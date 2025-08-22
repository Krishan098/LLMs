# summarization

'''
Summarization creates a shorter version of a document or an article that captures all the important information.
It can be formulated as a sequence-to-sequence task.
can be of two types:
- Extractive: extract the most relevant information from a document.
- Abstractive: generate new text that captures the most relevant information.
'''

from dotenv import load_dotenv
load_dotenv()
import os
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"),add_to_git_credential=True)
from datasets import load_dataset
billsum=load_dataset("billsum",split="ca_test")
billsum=billsum.train_test_split(test_size=0.2)
#print(billsum["train"][0])

'''
text: the text of the bill which'll be the input to the model.
summary: a condensed version of text which'll be the model target.
'''

from transformers import AutoTokenizer
checkpoint="google-t5/t5-small"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

'''
1. Prefix the input with a prompt so T5 knows this is a summarization task.
2. Use the keyword text_target argument when tokenizing labels.
3. Truncate sequences to be no longer than the maximum length set by the max_length parameter
'''

prefix="summarize: "

def preprocess_function(examples):
    inputs=[prefix + doc for doc  in examples["text"]]
    model_inputs=tokenizer(inputs,max_length=1024,truncation=True)
    labels=tokenizer(text_target=examples["summary"],max_length=128,truncation=True)
    model_inputs['labels']=labels["input_ids"]
    return model_inputs
tokenized_billsum=billsum.map(preprocess_function,batched=True)
from transformers import DataCollatorForSeq2Seq
data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=checkpoint)

import evaluate
rouge=evaluate.load("rouge")

import numpy as np

def compute_metrics(eval_pred):
    predictions,labels=eval_pred
    decoded_preds=tokenizer.batch_decode(predictions,skip_special_tokens=True)
    labels=np.where(labels!=-100,labels,tokenizer.pad_token_id)
    decoded_labels=tokenizer.batch_decode(labels,skip_special_tokens=True)
    result=rouge.compute(predictions=decoded_preds,references=decoded_labels,use_stemmer=True)
    prediction_lens=[np.count_nonzero(pred!=tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"]=np.mean(prediction_lens)
    return {k:round(v,4) for k,v in result.items()}

from transformers import AutoModelForSeq2SeqLM,Seq2SeqTrainingArguments,Seq2SeqTrainer
model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
training_args=Seq2SeqTrainingArguments(
    output_dir='billsum_model',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)
trainer=Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.push_to_hub()