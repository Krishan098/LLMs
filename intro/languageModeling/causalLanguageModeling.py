#Cauual Language Modeling

"""
Language modeling is used to predict next token in a sequence of tokens, and the model can only attend to the models on the left
Eg GPT-2

Metrics:
1. Perplexity: exponential of cross-entropy loss
2. Cross Entropy
"""

from dotenv import load_dotenv
load_dotenv()
import os
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"),add_to_git_credential=True)
from datasets import load_dataset
askscience=load_dataset("dany0407/eli5_category",split="train[:5000]")
askscience=askscience.train_test_split(test_size=0.2)


#print(askscience["train"][0])

from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("distilbert/distilgpt2")

askscience=askscience.flatten()
# creates a separate column for each subfield

# let's convert the list to a string so we can jointly tokenize them.

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])
tokenized_askscience=askscience.map(preprocess_function, batched=True,num_proc=1,remove_columns=askscience["train"].column_names,)

block_size=128
def group_texts(examples):
    concatenated_examples={k: sum(examples[k],[]) for k in examples.keys()}
    total_length=len(concatenated_examples[list(examples.keys())[0]])
    if total_length>=block_size:
        total_length=(total_length//block_size)*block_size
    result={
        k:[t[i:i+block_size] for i in range(0,total_length,block_size)] for k,t in concatenated_examples.items()
    }
    result['labels']=result["input_ids"].copy()
    return result
lm_dataset=tokenized_askscience.map(group_texts,batched=True,num_proc=1)

from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token=tokenizer.eos_token
data_collator=DataCollatorForLanguageModeling(tokenizer,mlm=False)

from transformers import AutoModelForCausalLM,TrainingArguments,Trainer

model=AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

training_args=TrainingArguments(
    output_dir="eli5_clm-model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True
)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,    
)

import math

eval_results=trainer.evaluate()
print(f"Perplexity:{math.exp(eval_results['eval_loss']):.2f}")
trainer.push_to_hub()