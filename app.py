import streamlit as st
from utils import extract_text_from_pdf, chunk_text, create_vector_store
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import tempfile
import os
from dotenv import load_dotenv
from pydantic import SecretStr
import time
import numpy as np

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set in your .env file.")
GROQ_API_KEY = SecretStr(api_key)

st.title("NoteMate – Your AI PDF Companion (Groq Edition)")

# Initialize chat history and stats in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'is_generating' not in st.session_state:
    st.session_state['is_generating'] = False
if 'timings' not in st.session_state:
    st.session_state['timings'] = []

# Upload and process PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.info("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    st.success("Text extracted!")

    st.info("Chunking text...")
    chunks = chunk_text(text)
    st.success(f"Chunked into {len(chunks)} pieces.")

    st.info("Creating vector store (may take a minute)...")
    vector_store = create_vector_store(chunks)
    st.success("Vector store ready!")

    st.session_state['vector_store'] = vector_store

# Only show Q&A if vector store is ready
if 'vector_store' in st.session_state:
    st.markdown("---")
    st.subheader("Ask questions about your PDF")

    # Custom input with arrow button
    col1, col2 = st.columns([8, 1])
    with col1:
        user_question = st.text_input(
            "Type your question:",
            key="user_question",
            label_visibility="collapsed"
        )
    with col2:
        submit = st.button("➡️")

    # If user presses arrow or Enter
    if submit and st.session_state['user_question']:
        st.session_state['is_generating'] = True
        with st.spinner('Hang there, generating answer...'):
            llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0)
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state['vector_store'].as_retriever()
            )

            # ⏱️ Measure query response time
            t0 = time.perf_counter()
            answer = qa.invoke(st.session_state['user_question'])
            t1 = time.perf_counter()

            response_time = t1 - t0
            st.write(f"⏱️ Response time: {response_time:.2f} seconds")

            # Store timings for summary stats
            st.session_state['timings'].append(response_time)

            answer_str = answer["result"]
            st.session_state['chat_history'].append({
                "question": st.session_state['user_question'],
                "answer": answer_str,
                "time": response_time
            })

        st.session_state['is_generating'] = False
        st.session_state.pop("user_question")
        st.markdown("<script>window.location.reload();</script>", unsafe_allow_html=True)

    # Display chat history
    for chat in reversed(st.session_state['chat_history']):
        st.markdown(f"""
        <div style='background-color:#1976D2; border-radius:12px; padding:1em; margin-bottom:1em; 
                    box-shadow: 0 2px 8px #0002; color:white;'>
            <b style='color:#FFC107;'>You:</b> {chat['question']}<br><br>
            <b style='color:#00E676;'>NoteMate:</b><br>
            <div style='font-size:1.1em;'>{chat['answer']}</div>
            <div style='font-size:0.9em;color:#ccc;'>⏱️ {chat['time']:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)

    # Show summary stats after multiple queries
    if len(st.session_state['timings']) > 3:
        arr = np.array(st.session_state['timings'])
        st.markdown("### 📊 Performance Summary")
        st.write(
            f"- Average response: {arr.mean():.2f}s\n"
            f"- Median response: {np.median(arr):.2f}s\n"
            f"- 95th percentile: {np.percentile(arr,95):.2f}s"
        )
