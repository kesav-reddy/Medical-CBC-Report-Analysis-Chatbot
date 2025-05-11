import streamlit as st
import openai
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Function to retrieve OpenAI API key
def get_openai_api_key(email):
    url = "http://52.66.239.27:8504/get_keys"
    payload = json.dumps({"email": email})
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        return response.json().get("api_key")
    else:
        st.error("Error retrieving API key.")
        return None

# Function to interact with OpenAI's ChatGPT
def chat_with_openai(prompt, api_key):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Using GPT-4
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Load the SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load embedded articles from the JSON file
with open('/Users/kesavreddy/Downloads/College/College/Actalyst/embedded_articles.json', 'r', encoding='utf-8') as f:
    embedded_articles = json.load(f)

# Streamlit UI setup
st.title("Chatbot with Embedded Articles")
email = "paderlakesavreddy@gmail.com"  # Your email for API key retrieval

# Retrieve OpenAI API key
api_key = get_openai_api_key(email)

if api_key:
    # Input for the user's message
    user_input = st.text_input("You: ", "")

    if user_input:
        # Find the most relevant embedded article (optional)
        user_embedding = embedding_model.encode(user_input)
        similarities = [np.dot(user_embedding, np.array(article['embedding'])) for article in embedded_articles]
        most_relevant_index = np.argmax(similarities)
        most_relevant_article = embedded_articles[most_relevant_index]

        # Prepare the prompt with the most relevant article
        prompt = f"{user_input}\n\nRelated Article:\nTitle: {most_relevant_article['title']}\nDescription: {most_relevant_article['description']}"

        # Generate a response using OpenAI
        response = chat_with_openai(prompt, api_key)
        st.text_area("ChatGPT: ", value=response, height=200)
else:
    st.warning("API key not retrieved. Please check your email.")
