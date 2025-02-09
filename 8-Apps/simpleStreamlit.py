import streamlit as st

from langchain_ollama import OllamaLLM

# Configure Ollama LLM
llm = OllamaLLM(
    model="llama3.2:latest",
    base_url="http://localhost:11434",
    temperature=0.1
)



st.title("Chat with AI")

# Initialize the language model

# User input
user_input = st.text_input("Ask me anything:")

if user_input:
    response = llm.invoke(user_input)
    st.write("AI Response:", response)

