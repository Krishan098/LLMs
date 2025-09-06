# TOKENIZER

"""
Tokenizers are the core components of the NLP pipeline.
They convert the text inputs to numerical data.
"""

## Types of tokenizers

### Word based
'''
The goal is to split the text into words and find a numerical representation
for each of them.

There are different ways to split the text. we can do it either on spaces or on punctuations too.
'''
## On spaces
tokenized_text="Jim Henson was a puppeteer.".split()
#print(tokenized_text)

## On Punctuation 

'''
If we wish to cover a language with a word-based tokenizer, we'll need to have an identifier for each word in the language
which generates a huge amount of tokens. Similar words will get a different mapping such as 'dog' and 'dogs'.
The model won't be able to see the similarity initially.

Finally we need a custom token to represent words that are not in our vocab. These are known as the "unknown" token, often represented
as "[UNK]" or "<unk>".

If we see a lot of these tokens being produced then it's a bad sign as the tokenizer wasn't able to retrieve a sensible representation of
a word and we're losing a lot of info along the way. The goal when crafting the vocabulary is to do it in such a way that the tokenizer 
tokenizes as few words as possible into the unknown token.
'''

## Character based tokenization

'''
splits tokens into characters, rather than words.
Benefits:
    - vocabulary is much smaller
    - There are much fewer out-of-vocabulary (unknown) tokens, since every word can be built from characters.
Problems:
    - less meaningful
    - we end up with a large number of tokens to be processed by our model
'''

## Subword tokenization

'''
They rely on the principle that frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords.
These subwords end up providing a lot of semantic meaning. 
'''

# Loading and Saving

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
#print(tokenizer("Using a Transformer network is simple"))
tokenizer.save_pretrained("/transformer/")

## Encoding

"""
Translating text to numbers is known as encoding
Done in 2 steps:
    - tokenization
    - conversion to Input IDs
"""

'''
After tokenization the next step is to convert the tokens into numbers, so we can build a 
tensor out of them and feed it to the model. To do this, the tokenizer has a vocabulary, which
is the part we download when we instantiate it with the from_pretrained() method.
'''
## tokenization

sequence="This is about to get crazier"

tokens=tokenizer.tokenize(sequence)
#print(tokens)

## Input to ids
ids= tokenizer.convert_tokens_to_ids(tokens)

#print(ids)


## Decoding 

'''
Decoding is going the other way around: from vocabulary indeices, we want to get a string.
'''

decoded_string=tokenizer.decode(ids)
#print(decoded_string)

""" it not only converts the indices back to tokens, but also groups together the tokens that were part of the same words to produce a readable sentence."""