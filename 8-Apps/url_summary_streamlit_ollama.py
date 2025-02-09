import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

from langchain_ollama import OllamaLLM


# Streamlit app
st.subheader('Summarize URL')


# Configure Ollama LLM
llm = OllamaLLM(
    model="llama3.2:latest",
    base_url="http://localhost:11434",
    temperature=0.1
)

# Get OpenAI API key and URL to be summarized
url = st.text_input("URL", label_visibility="collapsed")

# If 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not validators.url(url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Please wait..."):
                # Load URL data
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                data = loader.load()
                
                # Initialize the ChatOpenAI module, load and run the summarize chain
                #llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)
                prompt_template = """Write a summary of the following in 250-300 words:
                    
                    {text}

                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(data)

                st.success(summary)
        except Exception as e:
            st.exception(f"Exception: {e}")