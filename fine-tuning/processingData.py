import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
checkpoint="bert-base-cased"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences=[
    "I've been waiting for this course since a long time.",
    "This course is beautiful!"
]
batch=tokenizer(sequences,padding=True,truncation=True,return_tensors='pt')
#print(batch)
batch["labels"]=torch.tensor([1,1])
optimizer=AdamW(model.parameters())
loss=model(**batch).loss
loss.backward()
#print("loss=",loss)
optimizer.step()


# We'll use the MRPC dataset


from datasets import load_dataset

raw_datasets=load_dataset("glue","mrpc")
#print(raw_datasets)

'''
DatasetDict object contains the training set, the validation set and the test set.
'''

raw_train_dataset=raw_datasets["train"]
#print(raw_train_dataset[0])

#print(raw_train_dataset.features)

#print(raw_train_dataset[14])
val_dataset=raw_datasets["validation"]
#print("87th val entry ",val_dataset[86])
#print('sentence 1:',raw_datasets["train"]["sentence1"])
# tokenized_sentence_1=tokenizer(raw_datasets["train"]["sentence1"])
# tokenized_sentence_2=tokenizer(raw_datasets["train"]["sentence2"])

inputs=tokenizer("This is the first sentence.","This is the second one.")
#print(f"inputs:{inputs}")

'''token_type_ids:tells the model which part of the input is the first sentence and which 
one is the second'''

#print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
"""
[CLS] sentence1 [SEP] all have a token type ID of 0, while the other parts, corresponding to sentence2 [SEP], all have a token type ID of 1.
"""
'''
BERT is pretrained with token type IDs and on top of the masked language modeling objective, it has an additional objective called next sentence prediction.
The goal with this task is to model the relationship between pairs of sentences.
With next sentence prediction, the model is provided pairs of sentences(with randomly masked tokens) and asked to predict whether the second sentence follows the first.
To make the task non-trivial, half of the time the sentences follow each other in the original document they were extracted from, and the other half of the time the 2 sentences 
come from two different documents.
'''

#tokenized_dataset=tokenizer(raw_train_dataset,padding=True,truncation=True,)

def tokenize_function(example):
    return tokenizer(example["sentence1"],example["sentence2"],truncation=True)

tokenized_datasets=raw_datasets.map(tokenize_function,batched=True,num_proc=4)
print(tokenized_datasets)

'''DYNAMIC PADDING: pad all the examples to the length of the longest element when we batch elements together.'''

"""The function that is responsible for putting together samples inside a batch is called a collate function.
It's an argument we can pass when we build a DataLoader, the default being a function that will just convert our samples to PyTorch tensors and concatenate them.
"""

### TPUs prefer fixed shapes, even if it requires extra padding.



from transformers import DataCollatorWithPadding

data_collator=DataCollatorWithPadding(tokenizer=tokenizer)