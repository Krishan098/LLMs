# Token Classification
'''
Assigns a label to individual tokens in a sentence. 

Named Entity Recognition attempts to find a label for each entity in a sentence, such as a person,location or organization
'''
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()
import os
login(token=os.getenv("HF_TOKEN"),add_to_git_credential=True)
from datasets import load_dataset

wnut = load_dataset("ashwathjadhav23/wnut17_filtered_entities")

#print(wnut["train"][0])

label_list=wnut["train"].features[f"ner_tags"].feature.names
#print(label_list)

'''
['O', 'B-corporation', 'I-corporation', 'B-creative-work', 'I-creative-work', 'B-group', 'I-group', 'B-location', 'I-location', 'B-person', 'I-person', 'B-product', 'I-product']
The letter that prefixes each ner_tag indicates the token position of the entity:
B- indicates the beginning of an entity
I- indicates a token is contained inside the same entity
0 indicates the token doesn't correspond to any entity.
'''
 # preprocess
 
from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
example=wnut["train"][0]
tokenized_input=tokenizer(example["tokens"],is_split_into_words=True)
tokens=tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
#print(tokens)

'''
A single word corresponding to a single label may now be split into two subwords.
We'll need to realign the tokens and labels by:

1. Mapping all tokens to their corresponding word with the word_id method.

2. Assigning the label -100 to the special tokens [CLS] and [SEP] so they're ignored by the PyTorch loss function.

3. Only labeling the first token of a given word. Assign -100 to other subtokens from the same word.
'''

def tokenize_and_align_labels(examples):
    tokenized_inputs=tokenizer(examples["tokens"],truncation=True,is_split_into_words=True)
    labels=[]
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids=tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx=None
        label_ids=[]
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx!=previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx=word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"]=labels
    return tokenized_inputs
tokenized_wnut=wnut.map(tokenize_and_align_labels,batched=True)

from transformers import DataCollatorForTokenClassification

data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)

# Evaluate

import evaluate

labels=[label_list[i] for i in example[f"ner_tags"]]
import numpy as np

seqeval=evaluate.load('seqeval')
def compute_metrics(p):
    predictions,labels=p
    predictions=np.argmax(predictions,axis=2)
    true_predictions=[
        [label_list[p] for (p,l) in zip(prediction,label) if l!=-100] for prediction,label in zip(predictions,labels)]
    true_labels=[
        [label_list[l] for (p,l) in zip(prediction,label) if l!=-100] for prediction,label in zip(predictions,labels)]
    results=seqeval.compute(predictions=true_predictions,references=true_labels)
    return{
        "precision":results["overall_precision"],
        "recall":results["overall_recall"],
        "f1":results["overall_f1"],
        "accuracy":results["overall_accuracy"],
    }
    
# Train: before that create a map of the expected ids to their labels with id2label and label2id

id2label={
    0:"0",
    1:"B-corporation",
    2:"I-corporation",
    3:"B-creative-work",
    4:"I-creative-work",
    5:"B-group",
    6:"I-group",
    7:"B-location",
    8:"I-location",
    9:"B-person",
    10:"I-person",
    11:"B-product",
    12:"I-product",
}

label2id={
    "0":0,
    "B-corporation":1,
    "I-corporation":2,
    "B-creative-work":3,
    "I-creative-work":4,
    "B-group":5,
    "I-group":6,
    "B-location":7,
    "I-location":8,
    "B-person":9,
    "I-person":10,
    "B-product":11,
    "I-product":12,
}
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
classifier=AutoModelForTokenClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",num_labels=13,id2label=id2label,label2id=label2id
)

'''
1. Define your training hyperparameters in TrainingArguments
2. Pass the training arguments to Trainer along with model, dataset, tokenizer, data collator, and compute_metrics
3. call train
'''
trainingArgs=TrainingArguments(
    output_dir="TokenClassifierModel",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer=Trainer(
    model=classifier,
    args=trainingArgs,
    train_dataset=tokenized_wnut["train"],
    eval_dataset=tokenized_wnut["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()