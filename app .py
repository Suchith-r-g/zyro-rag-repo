import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢")
st.title("🏢 Zyro Dynamics HR Help Desk")

@st.cache_resource
def get_retriever():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    if not os.path.exists(data_dir):
        st.error(f"Cannot find the folder: {data_dir}. Did you upload it to GitHub?")
        st.stop()
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf", 
        loader_cls=PDFPlumberLoader
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200, add_start_index=True)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})
# Initialize LLM using Streamlit Secrets
llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])
retriever = get_retriever()

# Modern RAG chain setup
prompt = ChatPromptTemplate.from_template("""
Answer the user's question based on the context: {context}
Question: {input}
""")
document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)

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
            # Using modern chain invocation
            response = qa_chain.invoke({"input": prompt})
            result = response["answer"]
            st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
