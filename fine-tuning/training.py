'''
Trainer class helps us to fine-tune any of the pre-trained models it 
'''
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets=load_dataset('glue','mrpc')
checkpoint="bert-base-cased"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"],example["sentence2"],truncation=True)
tokenized_data=raw_datasets.map(tokenize_function,batched=True)
data_collator=DataCollatorWithPadding(tokenizer=tokenizer)

'''
The first step is to define a TrainingArguments class that contains all the hyperparameters the Trainer will use for training and evaluation.
'''

from transformers import TrainingArguments

training_args=TrainingArguments("test-trainer",push_to_hub=True)

from transformers import AutoModelForSequenceClassification

model=AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=2)

from transformers import Trainer

trainer=Trainer(
    model,
    training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data['validation'],
    data_collator=data_collator,
    processing_class=tokenizer, # when the processing_class is passed as tokenizer, the default data_collator used by the Trainer will be DataCollatorWithPadding.
)

trainer.train()

'''
won't tell about the model's performance because:
1. we didn't tell the Trainer to evaluate during training steps by setting eval_strategy in trainingArguments to either "steps"(evaluate every eval_steps) or epoch(evaluate at the end of each epoch)

2. We didn't provide a compute_metrics() function to calculate a metric during said evaluation
'''

predictions=trainer.predict(tokenized_data["vaidation"])

print(predictions.predictions.shape,predictions.label_ids.shape)


''' the output of predict() method is a tuple with 3 fields: predictions, label_ids and metrics.
Metrics contain the loss on the data passed,as well as some time metrics.

Predictions is a 2d array. to transform them into predictions that we can compare to our labels, we need to take the index with the maximum
value on the second axis'''

import numpy as np

preds=np.argmax(predictions.predictions,axis=-1)

import evaluate

metric = evaluate.load("glue","mrpc")
metric.compute(predictions=preds,references=predictions.label_ids)

def compute_metrics(eval_preds):
    metric=evaluate.load('glue','mrpc')
    logits,labels=eval_preds
    predictions=np.argmax(logits,axis=-1)
    return metric.compute(predictions=predictions,references=labels)

training_args=TrainingArguments(
    'test-trainer',
    eval_strategy="epoch",
    fp16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,# Effective batch size = 4*4=16
    lr_scheduler_type="cosine", # learning rate scheduling, default= linear
    )
model=AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=2)

trainer=Trainer(
    model,
    training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    )

trainer.train()

'''
    The Trainer API provides a high-level interface that handles most training complexity
    Use processing_class to specify your tokenizer for proper data handling
    TrainingArguments controls all aspects of training: learning rate, batch size, evaluation strategy, and optimizations
    compute_metrics enables custom evaluation metrics beyond just training loss
    Modern features like mixed precision (fp16=True) and gradient accumulation can significantly improve training efficiency
'''