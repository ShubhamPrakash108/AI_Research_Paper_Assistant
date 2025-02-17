import streamlit as st
import os
import base64
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import random

GEMINI_API_KEYS = [
    "AIzaSyCzTWJ5ZpkTGcnfabAXQjOSvMqrcwG7onk",
    "AIzaSyDqJBsNCbJEHp80nXTS4BzNOMD7e3KSQ10",
    "AIzaSyBPgJh_sRwKLAz0jR2EpAIy-F2zmKPR2qQ",
    "AIzaSyAYr05BjrCqRjNTrAvv2a50yPoe1Cpgw4A",
    "AIzaSyCchpkS1qxo2fqwT15H7BGd-41Tn-n9M24",
    "AIzaSyDfokL9HczAzRasWl_GNtT41E0gA2jkUJU",
    "AIzaSyD13qlRA23cAWjHPkb1uBaNdCUpBuPxoI0",
    "AIzaSyCRtKweOh3JPH8Gn1ACMrejuiQIW0N3RsY",
    "AIzaSyBlUyW5_eKud705mqzfo_athWNF-UgGlJY",
    "AIzaSyDxM-jtsl6DF4lm67lFWIumRXQACeZcHqo",
    "AIzaSyAteEW-4HLXkAudnDSKudFEyVP6rN9FT2c"
]



if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "document_metadata" not in st.session_state:
    st.session_state.document_metadata = {}

if "api_key_index" not in st.session_state:
    st.session_state.api_key_index = 0


def setup_vector_embeddings():
    
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=random.choice(GEMINI_API_KEYS)
    )
    
    all_documents = []
    all_metadata = {}
    
    folders = ["reference_research_papers", "target_RP"]
    for folder in folders:
        if os.path.exists(folder):
            pdf_loader = PyPDFDirectoryLoader(folder)
            documents = pdf_loader.load()
            
            for doc in documents:
                file_path = doc.metadata.get('source', '')
                if file_path:
                    file_name = os.path.basename(file_path)
                    all_metadata[file_name] = {'folder': folder, 'path': file_path}
            
            all_documents.extend(documents)
            st.write(f"Loaded {len(documents)} documents from {folder}")
    
    st.session_state.document_metadata = all_metadata
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(all_documents[:100])  
    
    faiss_store = FAISS.from_documents(split_documents, embedding_model)
    st.session_state.vector_store = faiss_store
    st.session_state.all_documents = all_documents
    
    return all_documents

def find_all_pdfs():
    pdf_files_dict = {}
    folders = ["reference_research_papers", "target_RP"]
    
    for folder in folders:
        if os.path.exists(folder):
            pdf_files = [f for f in os.listdir(folder) if f.endswith('.pdf')]
            for pdf in pdf_files:
                pdf_files_dict[pdf] = os.path.join(folder, pdf)
    
    return pdf_files_dict

def display_pdf(pdf_file):
    with open(pdf_file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.title("RAG Document Q&A With Google Gemini")

if st.button("Create Document Embedding"):
    documents = setup_vector_embeddings()
    st.write(f"VectorDB is ready with {len(documents)} total documents! 😍 Now you can ask questions.")

pdf_files_dict = find_all_pdfs()

if pdf_files_dict:
    pdf_options = [f"{pdf} (from {os.path.basename(os.path.dirname(path))})" 
                  for pdf, path in pdf_files_dict.items()]
    
    selected_pdf_option = st.selectbox("Select a PDF to view:", pdf_options)
    
    selected_pdf = selected_pdf_option.split(" (from ")[0]
    pdf_path = pdf_files_dict.get(selected_pdf)
    
    if st.button("Show PDF") and pdf_path:
        display_pdf(pdf_path)


def get_llm():
    
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=random.choice(GEMINI_API_KEYS)
    )

chat_prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant designed to answer questions based on the provided context. 
    Do not provide any additional information or assumptions outside the given context.
    <Context>
    {context}
    </Context>
    Question: {input}
    Provide a concise answer:
""")


user_query = st.text_input("Ask from LLM: ")

if user_query and "vector_store" in st.session_state:
    chat_llm = get_llm()
    document_chain = create_stuff_documents_chain(chat_llm, chat_prompt)
    retriever = st.session_state.vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    try:
        response_data = retrieval_chain.invoke({'input': user_query})
        st.session_state.conversation_history.append({"user": user_query, "response": response_data['answer']})
        st.write(response_data['answer'])
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        response_data = {"answer": f"Error: {str(e)}", "context": []}

    with st.expander("Conversation History"):
        for entry in st.session_state.conversation_history:
            st.write(f"**User:** {entry['user']}")
            st.write(f"**LLM:** {entry['response']}")

    history_text = "\n".join([f"User: {entry['user']}\nLLM: {entry['response']}\n" for entry in st.session_state.conversation_history])
    

    st.download_button(
        label="Download Conversation History",
        data=history_text,
        file_name="conversation_history.txt",
        mime="text/plain"
    )
    
    if "context" in response_data:
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response_data['context']):
                source = doc.metadata.get('source', 'Unknown source')
                st.write(f"**Source:** {source}")
                st.write(doc.page_content)
                st.write("---")
elif not "vector_store" in st.session_state and user_query:
    st.warning("Please create document embeddings first by clicking 'Create Document Embedding'")