# Behind the pipeline

from transformers import pipeline

classifier=pipeline('sentiment-analysis')

# print(classifier(
#     ["I've been waiting for a HuggingFace course my whole life to be a disaster.",
#      "You are always so nice to me,I hate this so much"]
# ))

'''
Pipeline groups these 3 steps together:
    - Preprocessing
    - passing the inputs through the model
    - postprocessing
'''
# preprocessing with a tokenizer 

'''
Like other neural networks, Transformer models can't process raw data directly, 
so the first step for the pipeline is to convert the text inputs into numbers 
that the model can make sense of.
To do this , it uses a tokenizer, which is responsible for:
 - splitting the input into words,subwords,symbols that are called tokens
 - mapping each token to an integer
 - Adding additional inputs that may be useful to the model
 
All this preprocessing needs to be done in the same way as when the model 
was pretrained, so we need to download that info from the model hub

We use the AutoTokenizer class for this and it's from_pretrained() method.
Using the checkpoint name of our model, it will automatically fetch the data 
associated with the model's tokenizer and cache it.
'''
from transformers import AutoTokenizer

checkpoint='distilbert-base-uncased-finetuned-sst-2-english'
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

'''
Once we get the tokenizer, we can pass our sentences directly to it and it would return a dict that 
could be fed to the model.
'''

'''
Now convert the list of inputs IDs to tensors
'''

'''Tensors are like Numpy arrays.
A numpy array can be scalar(0d), a vector(1d) or a matrix(2d), or have more dimensions.
'''

raw_inputs=["I've been waiting for a HuggingFace course my whole life to be a disaster.",
     "You are always so nice to me,I hate this so much"]
input=tokenizer(raw_inputs,padding=True,truncation=True,return_tensors="pt")
#print(input)
input_ids=input['input_ids']
#print(len(i) for i in input_ids)

'''
The output is a dict containing 2 keys: 
    - input_ids: contains 2 rows of integers(one for each sentence)
    - attention_mask:
'''

from transformers import AutoModel
model=AutoModel.from_pretrained(checkpoint)

'''
This architecture contains only the base Transformer module: given some inputs, it outputs hidden states, also known as features.
For each model input, we'll retrieve a high-dimensional vector representing the contextual understanding of that input by the Transformer model.

These hidden states are usually the inputs to another part of the model known as the head.
'''

# A high dimensional vector
'''
The vector output by the the Transformer module is usually large.It generally has 3 dimensions:
- Batch size: The number of sequences processed at a time.
- Sequence length: The length of the numerical representation of the sequence.
- Hidden size: The vector dimension of each model input.


It is said to be high dimensional because of the hidden size as it can be very large
'''

outputs=model(**input)
# print(outputs.last_hidden_state.shape)
# print(outputs)
# print(outputs[0])
# print(outputs["last_hidden_state"])


# Model heads

'''
The model heads take the high-dimensional vector of hidden states as input and project them onto a different dimension.
They are usually compposed of one or a few linear layers.

The output of the Transformer model is sent directly to the model head to be  processed.

The model is represented by it's embedding layers and the subsequent layers. The embedding layer converts each input ID in the 
tokenized input into a vector that represents the associated token. The subsequent layers manipulate those vectors using the attention 
mechanism to produce the final representation of the sentences.
''' 
from transformers import AutoModelForSequenceClassification
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs=model(**input)


#print(outputs.logits.shape)#->([2,2]) as 2 input sentences and 2 labels


# Postprocessing the output
#print(outputs.logits)
'''
logits: the raw, unnormalized scores outputted by the last layer of the model. 

all transformer models output the logits, as the loss function for training will generally 
fuse the last activation function, such as SoftMax, with the actual loss function, such as cross entropy.
'''

import torch 
predictions=torch.nn.functional.softmax(outputs.logits,dim=-1)
#print(predictions)

#get the label corresponding to each position using id2label

print(model.config.id2label)