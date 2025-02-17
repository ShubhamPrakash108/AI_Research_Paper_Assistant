import streamlit as st
import os
import base64
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
import google.generativeai as genai
import random
from googleapiclient.discovery import build
from typing import List, Dict

st.title("AI Research Companion🫂")

GEMINI_API_KEYS = ["AIzaSyCzTWJ5ZpkTGcnfabAXQjOSvMqrcwG7onk","AIzaSyDqJBsNCbJEHp80nXTS4BzNOMD7e3KSQ10","AIzaSyBPgJh_sRwKLAz0jR2EpAIy-F2zmKPR2qQ","AIzaSyAYr05BjrCqRjNTrAvv2a50yPoe1Cpgw4A","AIzaSyCchpkS1qxo2fqwT15H7BGd-41Tn-n9M24","AIzaSyDfokL9HczAzRasWl_GNtT41E0gA2jkUJU","AIzaSyD13qlRA23cAWjHPkb1uBaNdCUpBuPxoI0","AIzaSyCRtKweOh3JPH8Gn1ACMrejuiQIW0N3RsY","AIzaSyBlUyW5_eKud705mqzfo_athWNF-UgGlJY","AIzaSyDxM-jtsl6DF4lm67lFWIumRXQACeZcHqo","AIzaSyAteEW-4HLXkAudnDSKudFEyVP6rN9FT2c"]
YOUTUBE_API_KEY = "AIzaSyA_lI7cG4dWRspM7Sd2bVY_xmIXeZSSZJc"

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

class APIHandler:
    @staticmethod
    def get_gemini_response(prompt: str) -> str:
        try:
            api_key = random.choice(GEMINI_API_KEYS)
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            st.error(f"API error: {str(e)}")
            return f"Error: {str(e)}"

def generate_search_query(pdf_title, api_key):
    """Generate optimized search queries using Gemini"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')
    
    prompt = f"""
    Based on the research paper title '{pdf_title}', generate:
    1. A YouTube search query to find related educational videos and lectures
    2. A web search query to find related academic articles and blog posts
    Format: Return only two lines, first line for YouTube query, second line for web search
    """
    
    response = model.generate_content(prompt)
    queries = response.text.strip().split('\n')
    return queries[0], queries[1]

def fetch_youtube_videos(query):
    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=3
    )
    response = request.execute()
    return response.get("items", [])

def get_blog_recommendations(title: str) -> List[Dict]:
    try:
        prompt = f"""Generate exactly 3 blog article recommendations related to this research paper title: {title}
        Format your response exactly like this example, including the dashes:
        
        Title: Understanding Deep Learning Architectures
        URL: https://towardsdatascience.com/deep-learning-architectures
        Description: A comprehensive guide to modern neural network designs and their applications.
        ---
        Title: The Future of Machine Learning
        URL: https://machinelearningmastery.com/future-ml
        Description: Exploration of emerging trends in AI and their impact on research.
        ---
        Title: Practical Applications of Neural Networks
        URL: https://medium.com/practical-neural-networks
        Description: Real-world implementations and case studies of neural network applications.
        """
        
        response = APIHandler.get_gemini_response(prompt)
        if not response or response.startswith("Error"):
            st.error("Failed to get blog recommendations from Gemini")
            return []
        
        blog_entries = response.split('---')
        
        recommendations = []
        for entry in blog_entries:
            if entry.strip():
                try:
                    lines = [line.strip() for line in entry.strip().split('\n') if line.strip()]
                    blog = {}
                    for line in lines:
                        if line.startswith('Title:'):
                            blog['title'] = line.replace('Title:', '').strip()
                        elif line.startswith('URL:'):
                            blog['url'] = line.replace('URL:', '').strip()
                        elif line.startswith('Description:'):
                            blog['description'] = line.replace('Description:', '').strip()
                    if all(k in blog for k in ['title', 'url', 'description']):
                        recommendations.append(blog)
                except Exception as e:
                    st.error(f"Error parsing blog entry: {e}")
                    continue
        
        return recommendations
    except Exception as e:
        st.error(f"Blog recommendation error: {e}")
        return []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

GEMINI_API_KEYS = ["AIzaSyCzTWJ5ZpkTGcnfabAXQjOSvMqrcwG7onk","AIzaSyDqJBsNCbJEHp80nXTS4BzNOMD7e3KSQ10","AIzaSyBPgJh_sRwKLAz0jR2EpAIy-F2zmKPR2qQ","AIzaSyAYr05BjrCqRjNTrAvv2a50yPoe1Cpgw4A","AIzaSyCchpkS1qxo2fqwT15H7BGd-41Tn-n9M24","AIzaSyDfokL9HczAzRasWl_GNtT41E0gA2jkUJU","AIzaSyD13qlRA23cAWjHPkb1uBaNdCUpBuPxoI0","AIzaSyCRtKweOh3JPH8Gn1ACMrejuiQIW0N3RsY","AIzaSyBlUyW5_eKud705mqzfo_athWNF-UgGlJY","AIzaSyDxM-jtsl6DF4lm67lFWIumRXQACeZcHqo","AIzaSyAteEW-4HLXkAudnDSKudFEyVP6rN9FT2c"]

left_col, right_col = st.columns([0.55, 0.45])

with left_col:
    pdf_folder = 'reference_research_papers'
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    selected_pdf = st.selectbox('Select a PDF file:', pdf_files)
    
    st.markdown('<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)

    if selected_pdf:
        pdf_path = os.path.join(pdf_folder, selected_pdf)
        
        if st.button('Process PDF'):
            st.success(f'{selected_pdf} selected!')
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_documents = text_splitter.split_documents(documents)
                embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                st.session_state.vector_store = FAISS.from_documents(split_documents, embeddings)
                
                current_api_key = random.choice(GEMINI_API_KEYS)
                youtube_query, web_query = generate_search_query(selected_pdf.replace(".pdf", ""), current_api_key)
                
                st.session_state.youtube_query = youtube_query
                st.session_state.web_query = web_query
                st.session_state.pdf_title = selected_pdf.replace(".pdf", "")
                
                st.success("PDF processed and search queries generated!")
        
        with open(pdf_path, 'rb') as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'''
            <div style="margin-left: -250px;">
                <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700px" style="float:left; margin-top: 10px;"></iframe>
            </div>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        if 'youtube_query' in st.session_state:
            st.markdown("### 🎥 Related YouTube Videos")
            videos = fetch_youtube_videos(st.session_state.youtube_query)
            for video in videos:
                video_id = video["id"]["videoId"]
                video_title = video["snippet"]["title"]
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                
                st.markdown(f"**[{video_title}]({video_url})**")
                st.video(video_url)
            
            if 'pdf_title' in st.session_state:
                st.markdown("### 📚 Blog Recommendations")
                blog_recommendations = get_blog_recommendations(st.session_state.pdf_title)
                
                if blog_recommendations:
                    for blog in blog_recommendations:
                        st.markdown(f"""
                        **[{blog['title']}]({blog['url']})**
                        {blog['description']}
                        """)
                else:
                    st.warning("No blog recommendations could be generated at this time.")

with right_col:
    st.subheader("Chat with PDF")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-thinking-exp",
        temperature=0,
        google_api_key=random.choice(GEMINI_API_KEYS)
    )
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    prompt = st.chat_input("Ask about the PDF...")
    
    if prompt:
        if st.session_state.vector_store is None:
            st.error("Please process a PDF first!")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            retrieval_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=st.session_state.vector_store.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
            )
            
            response = retrieval_chain.invoke({
                "question": prompt,
                "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.chat_history]
            })
            
            answer_with_sources = f"{response['answer']}\n\n**Sources:**\n"
            sources = [f"Page {doc.metadata.get('page', 'Unknown')}" for doc in response["source_documents"]]
            answer_with_sources += "- " + "\n- ".join(sorted(set(sources)))
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer_with_sources})
            with st.chat_message("assistant"):
                st.markdown(answer_with_sources)


st.markdown("""
    <style>
        .main .block-container {
            max-width: 95%;
            padding: 1rem;
        }
        .stChatFloatingInputContainer {
            bottom: 20px;
            width: 100% !important;
        }
    </style>
""", unsafe_allow_html=True)