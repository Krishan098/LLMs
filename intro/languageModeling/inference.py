from transformers import pipeline
prompt = "Somatic hypermutation allows the immune system to"
lm=pipeline("text-generation",model='eli5_clm-model')
lm(prompt)