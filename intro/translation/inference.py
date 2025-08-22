text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."

from transformers import pipeline

translator=pipeline('translation_en_to_fr',model='opus_book_model')
translator(text)