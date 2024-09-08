from transformers import pipeline

classifier = pipeline('sentiment-analysis')
print('-----------------')
print(classifier(["I've been waiting for a huggingface course my whole life.",
                  "I hate this so much."]))

classifier = pipeline('zero-shot-classification')
print('-----------------')
print(classifier(
    "this course is about the transformers library",
    candidate_labels = ["eduction", "politics" ,"business"]))

classifier = pipeline('text-generation')
print('-----------------')
print(classifier("in this course, we will teach you how to"))

classifier = pipeline('text-generation' , model = 'distilgpt2')
print('-----------------')
print(classifier("in this course, we will teach you how to",
                 max_length = 30,
                 num_return_sequences = 5)
    )

classifier = pipeline('fill-mask')
print('-----------------')
print(classifier("this course will teach you all about <mask> models.",
                 top_k = 2)
    )

classifier = pipeline('ner', grouped_entities = True)
print('-----------------')
print(classifier("My name is Sylvain and I work at Hugging Face in Brooklyn."))

classifier = pipeline('question-answering')
print('-----------------')
print(classifier(
    question="where do i work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn."))

classifier = pipeline('summarization')
print('-----------------')
print(classifier("""
    My name is Sylvain and I work at Hugging Face in Brooklyn."""))

classifier = pipeline('translation',model='Helsinki-NLP/opus-mt-fr-en')
print('-----------------')
print(classifier("Ce cours est produit par Hugging Face."))
