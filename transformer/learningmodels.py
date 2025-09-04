# Models

## Creating a transformer
from huggingface_hub import login

import os
from dotenv import load_dotenv
load_dotenv()

login(token=os.getenv("HF_TOKEN"))
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
"""
This would download and cache the model data from the Hub. 
The checkpoint name corresponds to the specific architecture and weights.

"bert-base-cased": BERT with the basic architecture(12 layers, 768 hidden layers, 12 attention layers)
cased means it is important to distinguish between upper and lower case.
"""

"""
The AutoModel class and its associates are simple wrappers to fetch the appropriate model architecture.
It's an auto class meaning it will guess the appropriate model architecture for us and instantiate the 
correct model class.
"""

# from transformers import BertModel
# model=BertModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Loading and Saving

model.save_pretrained("/transformer/BERTtrained")

'''
This would save 2 files :
config.json: all necessary attributes needed to build the model architecture.also contains metadata, such as origin of checkpoint and version of transformers at time of saving the model. 
pytorch_model.bin: known as the state dictionary, contains all the model's weights.

Both the files work together. config file is needed to know about the model architecture, while the model weights are the parameters of the model.
'''

# We can reuse a saved model

model_reuse=AutoModelForMaskedLM.from_pretrained("/transformer/BERTtrained")

 # or in cli huggingface-cli login

model.push_to_hub("BERTmodelTrial")

# Encoding Text

'''
Transformer models handle text by turning the inputs into numbers.
'''

from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
encode_input=tokenizer('Hello! This might be it.')
#print(encode_input)

decoded=tokenizer.decode(encode_input['input_ids'])

'''The tokenizer adds special tokens -[CLS] and [SEP] required by the model'''

# Padding inputs
'''
If we ask the tokenizer to pad the inputs, it will make all sentences the same length by adding a special padding token to the sentences that are 
shorter than the longest one:
'''
encoded_inputs=tokenizer(["How are you?","I'm fine,thank you!"],padding=True,return_tensors='pt')
#print(encoded_inputs)

# Truncating inputs

'''The tensors might get too big to be processed by the model.'''
encoded_input = tokenizer(
    "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence.",
    truncation=True,
)
#print(encoded_input["input_ids"])

encoded_inputs=tokenizer(["How are you;>","I'm fine, thank you!"],padding=True,truncation=True,max_length=5,return_tensors='pt')
#print(encoded_inputs)

# Special tokens

'''
Special tokens are added to better represent the sentence boundaries, such as the beginning of the sentence([CLS]) or separator between sentences([SEP])
'''
encoded_input = tokenizer("How are you?")
#print("encoded input['input_ids']=",encoded_input["input_ids"])
#print(tokenizer.decode(encoded_input["input_ids"]))

# Tensors only accept rectangular shapes. 
encoded_sequence=encoded_inputs['input_ids']
import torch
model_inputs=torch.tensor(encoded_sequence)

output=model(model_inputs)
#print(output)