'''
Pipeline connects a model with its necessary processing and post processing steps, allowing
us to directly input any text and get an intelligible answer.
'''
from transformers import pipeline
classifier=pipeline("sentiment-analysis",model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
print("Task 1: sentiment analysis:",classifier(["I've been removed from the team.","This is so unreal"]))

"""
Three main steps involved when we pass text to a pipeline:
    - The text is preprocessed into a format the model can understand.
    - The preprocessed inputs are passed to the model.
    - The predictions of the model are post-processed, so we can understand them.
"""

# Text pipelines

"""
    - text-generation: Generate text from a prompt
    - text-classification: Classify text into predefined categories
    - summarization: Create a shorter version of a text while preserving key information
    - translation: Translate text from one language to another
    - zero-shot-classification: Classify text without prior training on specific labels
    - feature-extraction: Extract vector representations of text
"""

# Image pipelines

"""
    - image-to-text: Generate text descriptions of images
    - image-classification: Identify objects in an image
    - object-detection: Locate and identify objects in images
"""

#Audio pipeline
"""
    - automatic-speech-recognition: Convert speech to text
    - audio-classification: Classify audio into categories
    - text-to-speech: Convert text to spoken audio
"""

# Multimodal pipelines
"""
-     image-text-to-text: Respond to an image based on a text prompt
"""
