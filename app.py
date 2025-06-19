import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langchain API KEY
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful personal assistant who responds to user queries."),
    ("human", "{question}")
])

# Response generation function
def generate_response(question, model_name, temperature):
    llm = Ollama(model=model_name, temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Streamlit UI
st.title("🤖 Enhanced Q&A Chatbot with LLMs")

# Sidebar model selection
llm_model = st.sidebar.selectbox("🔍 Select a Model", ["llama3.1", "gemma2"])

# Sidebar temperature slider
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# User input section
st.write("### Ask your questions 👇")
user_input = st.text_input("You:")

# Display response
if user_input:
    response = generate_response(user_input, llm_model, temperature)
    st.write("**Assistant:**", response)
else:
    st.write("ℹ️ Please enter your question above.")
