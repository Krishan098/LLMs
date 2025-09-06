# Handling multiple sequences

## Models expect a batch of inputs
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint="distilbert-base-uncased-finetuned-sst-2-english"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence="I'v been waiting for this opportunity all my life"

tokens=tokenizer.tokenize(sequence)
ids=tokenizer.convert_tokens_to_ids(tokens)
input_ids=torch.tensor(ids)
#print("input_ids:",input_ids)
#model(input_ids)

tokenized_inputs=tokenizer(sequence,return_tensors='pt')
#print(tokenized_inputs["input_ids"])


output1= model(tokenized_inputs["input_ids"])
#print('logits with special tokens:',output1)
input_ids=torch.tensor([ids])
#print("input_ids:",input_ids)

output=model(input_ids)
#print('logits:',output.logits)

'''Batching is the act of sending multiple sentences through the model, all at once.'''

batched_ids=[ids,ids]
batched_input_ids=torch.tensor(batched_ids)
output_=model(batched_input_ids)
# print(output_.logits)

'''
Batching allows the model to work when you feed it multiple sentences.
When we try to batch together two sentences, they might be of different lengths.Therefore we use padding.
'''

## padding

batched_ids=[
    [200,200,200],
    [200,200]
]
'''
Padding ensures all our sentences have the same length by adding a special word called the padding token to the sentences with fewer values.
'''
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids=[[200,200,200]]
sequence2_ids=[[200,200]]
batched_ids=[
    [200,200,200],
    [200,200,tokenizer.pad_token_id],
]
# print(model(torch.tensor(sequence1_ids)).logits)
# print(model(torch.tensor(sequence2_ids)).logits)
# print(model(torch.tensor(batched_ids)).logits)

'''
We get different tensors when we process it as a single input and  batch it with another sentence.
This is because the key feature of Transformer models is attention layers that contextualize each token. These will 
take into account the padding tokens since they attend to all of the tokens of a sequence.
Therefore we use an attention mask so that the attention layers ignore the padding tokens.
'''


## Attention Masks

'''
Attention masks are tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s: 1s indicate the corresponding
tokens should not be attended to.
'''

batched_ids=[
    [200,200,200],
    [200,200,tokenizer.pad_token_id],
]
attention_mask=[
    [1,1,1],
    [1,1,0],
]
outputs=model(torch.tensor(batched_ids),attention_mask=torch.tensor(attention_mask))
#print(outputs.logits)


sequences=["I am happy huggingface happened",
           "what do i even write in the second part"]
# dic={}
# for i in sequences:
#     tokens=tokenizer.tokenize(i)
#     dic[i]=tokens
# print("tokens:",[dic[i] for i in dic.keys()])
# ids={}
# for i in dic:
#     ids[i]=tokenizer.convert_tokens_to_ids(dic[i])
    
# input_tokens=torch.tensor([i] for i in ids.values())
# print(input_tokens)

tokens_=tokenizer(sequences,return_tensors='pt',padding=True)
#print(tokens_)

#output=model(input_tokens)
#print(output)

output_model=model(tokens_['input_ids'])
print(output_model)

## Longer sequences
'''
there is a limit to the lengths of the sequences we can pass the models.
Most models handle sequences of up to 512 or 1024 tokens and will crash when 
asked to process longer sequences.

Solutions:
    - use a model with a longer supported sequence length
    - Truncate your sequences.
    
Models have different supported sequence lengths, and some specialize in handling very long sequences.
'''

# otherwise just use truncation by specifying the max_sequence_length parameter