# How do Transformers work

- All the Transformer models including GPT,BERT,Mistral,T5 etc have been trained as language models. They have been trained on large  amounts of raw text in a self-supervised fashion.

- Self-supervised learning is a type of training in which the objective is automatically computed from the inputs of the model.

- This type of model develops a statistical understanding of the language it has been trained on, but it's less useful for specific practical tasks. Because of this, the general pretrained model then goes through a process called _transfer learning_ or _fine tuning_.

- During this process, the model is fine-tuned in a supervised way- i.e using human-annotated labels - on a given task.

- An example of a task is predicting the next word in a sentence having read the n previous words. This is called _language modeling_ because the output depends on the past and present innputs, but not the future ones.

- Another example is _masked language modeling_, in which th model predicts a masked word in the sentence.

- crazy $CO_2$ emissions from training a LLM.

```
from huggingface_hub import HfApi
api=HfApi()
models=api.list_models(emissions_thresholds=(None,100),cardData=True)
```
## Transfer learning

- _Pretraining_ is the act of training a model from scratch: the weights are randomly initialized, and the training starts without any prior knowledge.

- _Fine-tuning_ on the other hand is the training done after a model has been pretrained. To perform fine-tuning, we first acquire a pretrained language model, then perform additional training with a dataset specific to your task. 

- The fine-tuning will only require a limited amount of data: the knowledge the pretrained model has acquired is "transferred", hence the term **Transfer learning**

## General Transformer Architecture

- The model is primarily composed of two blocks:
    - **Encoder:** The encoder recieves an input and builds a representation of it. This means that the model is optimized to acquire understanding from the input.

    - **Decoder**: The decoder uses the encoder's representation along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.

- Each of these can be used independently, depending on the task:
    - **Encoder-only models:** Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.
    - **Decoder-only models**: Good for generative tasks such as text generation.
    - **Encoder-decoder models or sequence-to-sequence models:** Good for generative tasks that require an input, such as translation or summarization.

## Attention layers

- A key feature os that transformers are built with special layers called attention layers. 

- This helps the model to pay specific attention to certain words in the sentence you passed it when dealing with representations of each word.

- A word by itself has a meaning, but that meaning is deeply affected by the context.

## The transformers architecture

- It was originally created for translation.

- During training, the encoder recieves inputs in a certain language, while the decoder recieves the same sentences in the desired target language. In the encoder, the attention layers can use all the words in a sentence.

- The decoder however , works sequentially and can only pay attention to the words in the sentece that it has already translated. 

- To speed things up during training(when the model has access to target sentences), the decoder is fed the whole target, but it is not allowed to use future words.

![alt text](images\image-4.png)

- The first attention layer in a decoder block pays attention to all inputs to the decoder, but the second attention layer uses the output of the encoder. It can thus access the whole input sentence to best predict the current word. 

- The attention mask can also be used in the encoder/decoder to prvent the model from paying attention to some special words.

## Architectures vs checkpoints

- **Architecture:** This is the skeleton of the model- the definition of each layer and each operation that happens within the model.

- **Checkpoints:** These are the weights that will be loaded in a given architecture.

- **Model:** This is an umbrella term that can mean both.
