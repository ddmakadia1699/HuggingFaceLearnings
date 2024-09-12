from transformers import pipeline

classifier = pipeline('sentiment-analysis')
print('-----------------')
print(classifier(["I've been waiting for a huggingface course my whole life.",
                  "I hate this so much."]))
