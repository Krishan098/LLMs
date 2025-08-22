# Question Answering
'''
Extractive:
task of extracting an answer in a given document. Question answering models take a context which is the document we want to search in,

The answer is not generated but extracted from the context. This type of question answering is called extractive

Metric:
1. Exact match: looks for exact match between the predicted answer and the correct answer.
2. F1 score: average of precision and recall
    True positive: Number of shared tokens between the prediction and the correct answer.
    False positive: Number of tokens in the predicted sequence, excluding the shared tokens.
    False Negative: Number of tokens in the correct answer, excluding the shared tokens
    
Abstractive:
generate an answer from the context that correctly answers the question.
'''

from dotenv import load_dotenv
load_dotenv
import os
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"),add_to_git_credential=True)
from datasets import load_dataset

squad=load_dataset('rajpurkar/squad_v2',split="train[:5000]")
squad=squad.train_test_split(test_size=0.2)

#print(squad["train"][0])

'''
answers: the starting location of the answer token and the answer text.
context: background information from which the model needs to extract the answer.
question: the question a model should answer.
'''

from transformers import AutoTokenizer

squad.remove_columns(['title','id'])

tokenizer=AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


# Preprocessing steps

'''
1. Some examples may have a very long context that exceeds the maximum input length of the model. truncate only the context.

2. Map the start and end positions of the answer to the original context by setting return_offset_mapping=True.

3. With the mapping in hand, we can find the start and end tokens of the answer. Use the sequence_ids method to find which part of the offset corresponds to the question and which corresponds to the context.
'''

def preprocess_function(examples):
    questions=[q.strip() for q in examples['question']]
    inputs=tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping=inputs.pop("offset_mapping")
    answers=examples["answers"]
    start_positions=[]
    end_positions=[]
    for i,offset in enumerate(offset_mapping):
        answer=answers[i]
        if len(answer["answer_start"])==0:
            start_positions.append(0)
            end_positions.append(0)
            continue
        start_char=answer["answer_start"][0]
        end_char=answer["answer_start"][0]+len(answer["text"][0])
        sequence_ids=inputs.sequence_ids(i)
        idx=0
        while sequence_ids[idx]!=1:
            idx+=1
        context_start=idx
        while sequence_ids[idx]==1:
            idx+=1
        context_end=idx-1
        
        # if the answer is not fully inside the context, label it (0,0)
        
        if offset[context_start][0]>end_char or offset[context_end][1]<start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx=context_start
            while idx<=context_end and offset[idx][0]<=start_char:
                idx+=1
            start_positions.append(idx-1)
            
            idx=context_end
            while idx >=context_start and offset[idx][1]>=end_char:
                idx-=1
            end_positions.append(idx+1)
    inputs["start_positions"]=start_positions
    inputs["end_positions"]=end_positions
    return inputs
tokenized_squads=squad.map(preprocess_function,batched=True,remove_columns=squad["train"].column_names)
from transformers import DefaultDataCollator
data_collator=DefaultDataCollator()

from transformers import AutoModelForQuestionAnswering,TrainingArguments,Trainer
model=AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")

training_args=TrainingArguments(
    output_dir="qa_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True
)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squads["train"],
    eval_dataset=tokenized_squads["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)
trainer.train()
trainer.push_to_hub()