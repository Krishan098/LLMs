text="The Golden State Warriors are an American professional basketball team based in San Francisco"
from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv
import os
login(token=os.getenv("HF_TOKEN"),add_to_git_credential=True)
classifier= pipeline("ner",model="TokenClassifierModel",token=os.getenv("HF_TOKEN"))
print(classifier(text))
