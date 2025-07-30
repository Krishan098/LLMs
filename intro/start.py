'''
Pipeline connects a model with its necessary processing and post processing steps, allowing
us to directly input any text and get an intelligible answer.
'''
from transformers import pipeline
classifier=pipeline("sentiment-analysis",model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
#print("Task 1: sentiment analysis:",classifier(["I've been removed from the team.","This is so unreal"]))

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

# Zero-shot classification

zero_shot_classifier=pipeline("zero-shot-classification",model='facebook/bart-large-mnli')
resultZero=zero_shot_classifier("This is a heist supposed to be done on Monday.",candidate_labels=["education","suspense","thrill","crime","adventure"],)
#print(f"zero_shot: {resultZero}")

# Text generation

generator=pipeline("text-generation",model='openai-community/gpt2')
generation=generator("Life is worth it because",num_return_sequences=5,max_length=40)
#print(f"Task 3 : text generation:{generation}")

# Mask filling: fills in the blanks 

unmasker=pipeline("fill-mask",model='google-bert/bert-base-uncased')
filled=unmasker("This is a pretty [MASK] woman",top_k=2)
#print(f"Task: fill mask: {filled}")
"""
The top_k argument controls how many possibilities we want to display.
<mask>: mask token
"""

# Named Entity Recognition

"""
It is a task where the model has to find which parts of the input text correspond to entities such as persons, locations or organizations.
"""
ner= pipeline("ner",grouped_entities=True)
#print(ner("My name is Sylvain and I want to work at huggingFace."))

# Question answering

qa=pipeline("question-answering",model="distilbert/distilbert-base-cased-distilled-squad")
#print(qa(
#    question='What do I do?',
#    context="I am a Googler in Delhi"
#))

summary=pipeline("summarization",model="facebook/bart-large-cnn")
result=summary("""
               America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
               """)
#print(result)

# Translation
translator=pipeline("translation",model="Helsinki-NLP/opus-mt-fr-en")
# translator("eu come uma maca")

# Image classification

image_classifier=pipeline(task="image-classification",model="google/vit-base-patch16-224")
result=image_classifier("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pioeline-cat-chonk.jpeg")
print(result)

# Automatic speech recognition

transcriber=pipeline(task="automatic-speech-recognition",model="openai/whisper-large-v3")
result_transcribe=transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(result_transcribe)