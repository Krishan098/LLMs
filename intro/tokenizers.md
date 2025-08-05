# Tokenizers

- tokenizing a text is splitting it into words or subwords, which then are converted to ids through a look-up table. 

## Introduction

- A simple way of tokenizing this text is to split by spaces.

- This includes the punctuation too which is suboptimal. We should take the punctuation into account so that a model doesnot have to learn a different representation of a word and every possible punctuation symbol that could follow it, it explodes the number of representation the model has to learn.

- "Don't". "Don't" stands for "do not", so it would be better tokenized as ["Do", "n't"]. This is where things start getting complicated, and part of the reason each model has its own tokenizer type.

- A pretrained model only performs properly if you feed it an input that was tokenized with the same rules that were used to tokenize its training data.

- A big vocabulary size forces the model to have an enormous embedding matrix as the input and output layer, which causes both an increased memory and time complexity. 


### Character-based tokenization

- While character tokenization is very simple and would greatly reduce memory and time complexity it makes it much harder for the model to learn meaningful input representations.

- leads to a loss of performance.

- Therefore, transformers models use a hybrid between word-level and character level tokenization called **subword** tokenization.

### Subword tokenization

- Subword tokenization algorithms rely on the principle that frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords. 

- For instance "annoyingly" might be considered a rare word and could be decomposed into "annoying" and "ly". Both "annoying" and "ly" as stand-alone subwords would appear more frequently while at the same time the meaning of "annoyingly" is kept by the composite meaning of "annoying" and "ly".

- Subword tokenization allows the model to have a reasonable vocabulary size while being able to learn meaningful context-independent representations. In addition, subword tokenizations enables the model to process words it has never seen before, by decomposing them into known subwords.

### Byte-Pair Encoding 

- Byte-pair encoding(BPE) relies on a pre-tokenizer that splits the training data into words. Pretokenization can be as simple as space tokenization. More advanced pre-tokenization include rule-based tokenization,eg. XLM,FlauBERT which uses _Moses_ for most languages, or GPT which uses spaCy and ftfy, to count the frequency of each word in the training corpus.

- After pre-tokenization, a set of unique words has been created and the frequency with which each word occured in the training data has been determined. Next, BPE creates a base vocabulary consisting of all symbols that occur in the set of unique words and learns merge rules to form a new symbol from two symbols of the base vocabulary.It does so until the vocabulary size is a hyperparameter to define before training the tokenizer.

```("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)```

- Consequently, the base vocabulary is ["b", "g", "h", "n", "p", "s", "u"]. Splitting all words into symbols of the base vocabulary, we obtain:

```("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)```


- BPE then counts the frequency of each possible symbol pair and picks the symbol pair that occurs most frequently. 

- In the example above "h" followed by "u" is present 10 + 5 = 15 times (10 times in the 10 occurrences of "hug", 5 times in the 5 occurrences of "hugs"). However, the most frequent symbol pair is "u" followed by "g", occurring 10 + 5 + 5 = 20 times in total. Thus, the first merge rule the tokenizer learns is to group all "u" symbols followed by a "g" symbol together. Next, "ug" is added to the vocabulary. The set of words then becomes

```("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)```

- BPE then identifies the next most common symbol pair. It’s "u" followed by "n", which occurs 16 times. "u", "n" is merged to "un" and added to the vocabulary. The next most frequent symbol pair is "h" followed by "ug", occurring 15 times. Again the pair is merged and "hug" can be added to the vocabulary.

- At this stage, the vocabulary is ["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"] and our set of unique words is represented as

```("hug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("hug" "s", 5)```

- The learned merge rules would then be applied to new words. It would tokenize symbols that are not in the base vocabulary as "<unk>". 

### Byte-level BPE

- A base vocabulary that includes all possible base characters can be quite large if e.g. all unicode characters are considered as base characters. To have a better base vocabulary, GPT-2 uses bytes as a base vocabulary, which forces the base vocabulary to be of size 256 while ensuring that every base character is included in the vocabulary. 

### WordPiece

- It is the subword tokenization algorithm used for BERT,DistilBERT and Electra. 

- First, initializes the vocabulary to include every character present in the training data and progressively learns a given number of merge rules. In contrast to BPE, WordPiece doesnot choose the most frequent symbol pair, but the one that maximizes the likelihood of the training data once added to the vocabulary.

- Maximizing the likelihood of the training data is equivalent to finding the symbol pair, whose probability divided by the probabilities of its first symbol followed by its second symbol is the greatest among all symbol pairs. 

- E.g. "u", followed by "g" would have only been merged if the probability of "ug" divided by "u", "g" would have been greater than for any other symbol pair.

### Unigram 

- It initializes its base vocabulary to a large number of symbols and progressively trims down each symbol to obtain a smaller vocabulary. The base vocabulary could for instance correspond to all pre-tokenized words and the most common substrings. Unigram is not used directly for any of the models in the transformers, but it's used in conjunction with SentencePiece.

- At each training step, the Unigram algorithm defines a loss(often defined as the log-likelihood) over the training data given the current vocabulary and a unigram language model. Then, for each symbol in the vocabulary, the algorithm computes how much the overall loss would increase if the symbol was to be removed from the vocabulary.

- Unigram then removes p(with p usually being 10% or 20%) percent of the symbols whose loss increase is the lowest, i.e. the symbols that least affect the overall loss over the training data. This process is repeated until the vocabulary has reached the desired size. The Unigram algorithm always keeps the base characters so that any word can be tokenized.

- Because Unigram is not based on merge rules, the algorithm has several ways of tokenizing new text after training.

- Unigram saves the probability of each token in the training corpus on top of saving the vocabulary so that the probability of each possible tokenization can be computed after training. The algorithm simply picks the most likely tokenization in practice, but also offers the possibility to sample a possible tokenization according to their probabilities.

- Those probabilities are defined by the loss the tokenizer is trained on. Assuming that the training data consists of the words $x_1,\ldots,x_{\text{N}}$ and that the set of all possible tokenizations for a word $x_{\text{i}}$ is defined as S($x_{\text{i}}$), then the overall loss is defined as

$$
L=-\sum_{\text{i=1}}^{\text{N}} log(\sum_{text{x\epsilon S(x_{\text{i}})}}p(x))
$$

### SentencePiece

- All tokenization algorithms described so far have the same problem: it is assumed that the input text uses spaces to separate words. However not all languages use spaces to separate words.

- The XLNetTokenizer uses SentencePiece. Decoding with SentencePiece is very easy since all tokens can just be concatenated and "__" is replaced by a space.

 