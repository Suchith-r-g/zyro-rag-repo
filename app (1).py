import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢")
st.title("🏢 Zyro Dynamics HR Help Desk")

# Load and process data (cached for performance)
@st.cache_resource
def get_retriever():
    # Ensure your PDFs are in a folder named 'data'
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# Initialize LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])
retriever = get_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about HR policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            response = qa_chain.invoke(prompt)
            result = response["result"]
            st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
