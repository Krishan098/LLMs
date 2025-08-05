from transformers import BertTokenizer # sub-word tokenizer
tokenizer=BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
print(tokenizer.tokenize("This is a sub-word tokenizer."))

from transformers import XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained("xlnet/xlnet-base-cased")
print(tokenizer.tokenize("Don't you love 🤗 Transformers? We sure do."))

