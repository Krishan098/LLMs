from transformers import AutoTokenizer
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
tokenizer=AutoTokenizer.from_pretrained("text-classifier")
inputs=tokenizer(text,return_tensors="pt")
import torch
from transformers import AutoModelForSequenceClassification
model=AutoModelForSequenceClassification.from_pretrained("text-classifier")
with torch.no_grad():
    logits=model(**inputs).logits
    
predicted_class_ids=logits.argmax().item()
model.config.id2label[predicted_class_ids]