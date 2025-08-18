'''
Automatic Speech Recognition converts a speech signal to text, mapping a sequence of audio inputs to text outputs. 
'''
from datasets import load_dataset, Audio
from huggingface_hub import login

import os
from dotenv import load_dotenv
load_dotenv()
login(token=os.getenv("HUGGINGFACE_API_KEY"),add_to_git_credential=True)
minds=load_dataset("PolyAI/minds14",name="en-US",split="train[:100]")
minds=minds.train_test_split(test_size=0.2)
#print(minds)
minds=minds.remove_columns(['english_transcription','intent_class','lang_id'])
#print(minds["train"][0])

# PREPROCESS

from transformers import AutoProcessor

processor= AutoProcessor.from_pretrained("facebook/wav2vec2-base")
# need to sample the dataset to 16000Hz from 8000Hz.
minds=minds.cast_column("audio",Audio(sampling_rate=16_000))
#print(minds["train"][0])

# The Wav2Vec2 tokenizer is only trained on uppercase characters so we'll need to make sure the text matches the tokenizer's vocabulary.

def uppercase(text):
    return {"transcription":text["transcription"].upper()}
minds=minds.map(uppercase)

# Now let's create a preprocessing function that:
#    1. Calls the audio column to load and resample the audio file.
#    2. Extracts the input_values from the audio file and tokenize the transcription column with the processor.

def prepare_dataset(batch):
    audio=batch["audio"]
    batch=processor(audio["array"],sampling_rate=audio["sampling_rate"],text=batch["transcription"])
    batch["input_length"]=len(batch["input_values"][0])
    return batch

if __name__ == "__main__":
    encoded_minds=minds.map(prepare_dataset,remove_columns=minds.column_names["train"],num_proc=4)
else:
    encoded_minds=minds.map(prepare_dataset,remove_columns=minds.column_names["train"])

'''Data collators are objects that will form a batch by using a list of dataset elements as input. 
These elements are of the same type as the elements of train or eval dataset. To build batches they may apply padding.
Data collator with padding also dynamically pads the text and labels to the length of the longest element in its batch so they are a uniform length.'''

import torch
from dataclasses import dataclass, field
from typing import Any, Optional, Union
@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding:Union[bool,str]="longest"
    def __call__(self,features:list[dict[str,Union[list[int],torch.tensor]]])->dict[str,torch.tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features=[{"input_values":feature["input_values"][0]} for feature in features]
        label_features=[{"input_ids":feature["labels"]} for feature in features]
        batch=self.processor.pad(input_features,padding=self.padding,return_tensors="pt")
        labels_batch=self.processor.pad(labels=label_features,padding=self.padding,return_tensors="pt")
        labels=labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1),-100)
        batch["labels"]=labels
        return batch
data_collator=DataCollatorCTCWithPadding(processor=processor,padding="longest")

# Evaluate

import evaluate
'''# Word error rate is a common metric of the performance of an automatic speech recognition system.
# The general difficulty of measuring performance lies in the fact that the recognized word sequence can have a different length from the reference word sequence
The WER is derived from the Levenshtein distance, working at the word level instead of the phoneme level. 

WER= (S+D+I)/N=(S+D+I)/(S+D+C);
where S is the number of substitutions, D is the number of deletions, I is the number of insertions, C is the number of correct words, N is the number of words in the reference(N=S+D+C)
'''
wer=evaluate.load("wer")

import numpy as np

def compute_metrics(pred):
    pred_logits=pred.predictions
    pred_ids=np.argmax(pred_logits,axis=-1)
    pred.label_ids[pred.label_ids==-100]=processor.tokenizer.pad_token_id
    pred_str=processor.batch_decode(pred_ids)
    label_str=processor.batch_decode(pred.label_ids,group_tokens=False)
    wer=wer.compute(predictions=pred_str,references=label_str)
    return {"wer":wer}

from transformers import AutoModelForCTC, TrainingArguments, Trainer
model=AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

training_args=TrainingArguments(
    output_dir="asr_mind_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    gradient_checkpointing=True,
    max_steps=2000,
    fp16=True,
    group_by_length=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
if __name__=="__main__":
    trainer.train()

    trainer.push_to_hub()