import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# UI setup
st.set_page_config(page_title="TEK-UP Chatbot", layout='wide')
st.title("TEK-UP Website Chatbot")

# Model selection
MODEL_OPTIONS = {
    "LLaMa 3.3 (70B)": "llama-3.3-70b-versatile",
    "Gemma 2 (9B)": 'gemma2-9b-it',
    "LLaMa 3 (8B)": 'llama3-8b-8192',
    'Qwen (32B)': 'qwen-qwq-32b'
}
selected_model_label = st.selectbox("Choose a model", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[selected_model_label]

# Load pre-populated vector DB
@st.cache_resource
def load_vdb():
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    return FAISS.load_local("vdb", embeddings, allow_dangerous_deserialization=True)

vdb = load_vdb()

if 'history' not in st.session_state:
    st.session_state.history = [{
        'role': 'assistant',
        'content': 'Hi, I\'m Thabet, how can I help you ?'
    }]

# Input
user_input = st.chat_input("Ask a question about TEK-UP...")

# Handle input
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.spinner("Answering..."):
        retriever = vdb.as_retriever()
        llm = ChatGroq(api_key=GROQ_API_KEY, model=selected_model)

        relevant_docs = retriever.get_relevant_documents(user_input)
        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        query = f"""
        You are thabet, TEK-UP University Assistant. Answer the questions depending on the context.

        Context:
        {context}

        Question: {user_input}
        
        Answer in a respectful way, Be explicit and precise in your answers
        Always be kind and welcoming
        Always include up to date information do not give unnecessary information
        Do not give outdated informations
        Always answer the question without giving extra information that are not related to the question
        You are a helpful assistant
        You can be funny and loud sometimes
        """
        result = llm.invoke(query)
        st.session_state.history.append({"role": "assistant", "content": result.content})

# Display messages
for message in st.session_state.history:
    if message['role'] != 'system':
        with st.chat_message(message['role']):
            st.markdown(message['content'])
