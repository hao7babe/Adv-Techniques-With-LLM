import openai
from transformers import pipeline
import os

# Part 1: Translation, Summarization, Sentiment Analysis
def translate_summarize_sentiment(text):
    translator = pipeline('translation', model='Helsinki-NLP/opus-mt-fr-en')
    summarizer = pipeline('summarization')
    sentiment_analyzer = pipeline('sentiment-analysis')

    # Step 1: Translate
    translated_text = translator(text, max_length=512)[0]['translation_text']
    
    # Step 2: Summarize
    summary = summarizer(translated_text, max_length=60, min_length=30, do_sample=False)[0]['summary_text']
    
    # Step 3: Sentiment Analysis
    sentiment = sentiment_analyzer(summary)[0]
    
    return translated_text, summary, sentiment

french_text = """J'apprécie le sérieux avec lequel ce restaurant prend les allergies alimentaires. 
En tant que personne allergique aux noix, je me sentais complètement en sécurité en dînant ici. 
De plus, leurs options sans gluten et végétaliennes ont été une agréable surprise. Fortement recommandé."""

translated, summary, sentiment = translate_summarize_sentiment(french_text)
print("Translated:", translated)
print("Summary:", summary)
print("Sentiment:", sentiment)

# Part 2: GPT-4 Queries with Temperature
openai.api_key = os.getenv('OPENAI_API_KEY')

def query_gpt4(prompt, temperature):
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        temperature=temperature,
        max_tokens=100
    )
    return response.choices[0].text.strip()

prompt = "a patient has a low blood pressure and a high heart rate. just give me names of three most probable diseases without any other word."

for temp in [0.1, 1.4]:
    print(f"Results with temperature {temp}:")
    for i in range(5):
        print(f"Attempt {i+1}: {query_gpt4(prompt, temp)}")
