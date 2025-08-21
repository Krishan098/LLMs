import whisper

model=whisper.load_model("tiny")
import os
from dotenv import load_dotenv
load_dotenv()
from datasets import load_dataset
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))

audio=load_dataset("PolyAI/minds14",name="en-US",split="train[:100]")
print(audio[0])