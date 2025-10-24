import streamlit as st
import tempfile
import os
import time
import numpy as np
from dotenv import load_dotenv
from pydantic import SecretStr
from typing import List
from utils import extract_text_from_pdf, chunk_text, create_vector_store
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set in your .env file.")
GROQ_API_KEY = SecretStr(api_key)

# ---------- Page config ----------
st.set_page_config(
    page_title="Askly – Knowledge-based Search Engine",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom dark futuristic styles ----------
st.markdown(
    """
    <style>
    /* Base */
    html, body, [class*="stApp"], .main {
        background: linear-gradient(180deg,#04060a 0%, #0b0d12 40%, #071026 100%);
        color: #cbd5e1;
        font-family: 'Poppins', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* Neon header */
    .lm-header {
        background: linear-gradient(90deg, rgba(103,58,183,0.95), rgba(33,150,243,0.9));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Cards */
    .lm-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 6px 24px rgba(2,6,23,0.6), inset 0 1px 0 rgba(255,255,255,0.01);
    }

    .lm-card .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #e6eef8;
        margin-bottom: 8px;
    }

    /* Upload area */
    .upload-area {
        border: 1px dashed rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 12px;
        padding: 8px 14px;
        font-weight: 600;
        box-shadow: 0 6px 18px rgba(15, 34, 76, 0.45);
    }

    /* Chat bubble */
    .lm-bubble-user {
        background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
        color: #9fb7ff;
        padding: 12px;
        border-radius: 12px;
        border-left: 4px solid rgba(33,150,243,0.6);
    }
    .lm-bubble-assistant {
        background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        color: #b7f7d6;
        padding: 12px;
        border-radius: 12px;
        border-left: 4px solid rgba(103,58,183,0.6);
    }

    /* tiny copy feedback */
    .copy-feedback {
        color: #9be7ff;
        opacity: 0;
        transition: opacity .25s ease-in-out;
    }

    /* small metadata */
    .muted {
        color: rgba(255,255,255,0.45);
        font-size: 12px;
    }

    /* make file list readable */
    .file-name {
        color: #cfe9ff;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
col1, col2 = st.columns([8, 2])
with col1:
    st.markdown("<h1 class='lm-header'>Askly – Knowledge-based Search Engine</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Upload documents, build a vector store, and ask Askly questions. Dark Futuristic UI.</div>", unsafe_allow_html=True)
with col2:
    # Simple sidebar-like controls but in header
    theme = st.selectbox("Theme preset", ["Dark Futuristic"], disabled=True)
    st.markdown("<div class='muted' style='text-align:right;'>Status: <b style='color:#9be7ff;'>Ready</b></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------- Sidebar: quick stats & controls ----------
with st.sidebar:
    st.markdown("### ⚙️ Askly Controls")
    st.write("Upload multiple PDFs, rebuild the index, and manage memory.")
    rebuild_btn = st.button("🔁 Rebuild index")
    clear_sessions = st.button("🗑️ Clear session")

if clear_sessions:
    for k in list(st.session_state.keys()):
        st.session_state.pop(k, None)
    st.experimental_rerun()

# ---------- Initialize session state ----------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "is_generating" not in st.session_state:
    st.session_state["is_generating"] = False
if "timings" not in st.session_state:
    st.session_state["timings"] = []
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

# ---------- Upload & Processing Card ----------
st.markdown("<div class='lm-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📁 Upload PDFs (multiple)</div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Drop PDFs here or click to browse",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple PDF files. Askly will merge the extracted text into one index."
)

col_upload_1, col_upload_2 = st.columns([3,1])
with col_upload_1:
    if uploaded_files:
        st.markdown("<div class='upload-area'>", unsafe_allow_html=True)
        st.markdown("<b>Files to process:</b>", unsafe_allow_html=True)
        for f in uploaded_files:
            st.markdown(f"<div class='file-name'>• {f.name} — <span class='muted'>{f.size/1024:.1f} KB</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No files selected yet. Use the uploader above to add PDFs.")
with col_upload_2:
    process_btn = st.button("Search")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Process uploaded files ----------
def process_pdf_files(file_objs: List[st.runtime.uploaded_file_manager.UploadedFile]):
    """Save uploaded st files to temp files, extract text, chunk, and create vector store."""
    combined_texts = []
    file_count = len(file_objs)
    progress = st.progress(0)
    for idx, f in enumerate(file_objs):
        progress_text = f"Extracting text from {f.name} ({idx+1}/{file_count})..."
        st.info(progress_text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        extracted = extract_text_from_pdf(tmp_path)
        combined_texts.append((f.name, extracted))
        # small progress step
        progress.progress(int(((idx+1)/file_count)*100))
        # clean up temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    progress.empty()
    st.success(f"Extracted text from {file_count} files.")
    # Combine texts with small separators (retain filename metadata)
    merged_text = "\n\n".join([f"---\nSource: {name}\n---\n{text}" for name, text in combined_texts])
    st.info("Chunking combined text...")
    chunks = chunk_text(merged_text)
    st.success(f"Chunked into {len(chunks)} pieces.")
    st.info("Creating vector store (this may take a short while)...")
    vstore = create_vector_store(chunks)
    st.success("Vector store ready.")
    # update session
    st.session_state["vector_store"] = vstore
    # store list of files processed
    st.session_state["uploaded_files"] = [f.name for f in file_objs]
    return vstore

if process_btn and uploaded_files:
    try:
        # process immediately
        process_pdf_files(uploaded_files)
    except Exception as e:
        st.error(f"Error while processing files: {e}")

# Rebuild index if asked and we have previous uploaded files in session
if rebuild_btn and st.session_state.get("uploaded_files"):
    st.info("Rebuilding vector store from previously uploaded files...")
    # note: we don't have the original UploadedFile objects anymore; ask user to re-upload
    st.warning("Please re-upload the original PDF files into the uploader to rebuild the index.")
    # we don't auto-rebuild because Streamlit's UploadedFile objects are ephemeral

# ---------- Q&A / Interaction Card ----------
st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
st.markdown("<div class='lm-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>💬 Ask Askly</div>", unsafe_allow_html=True)

def create_retrieval_chain(retriever, llm):
    """
    Create a small adapter around RetrievalQA that exposes an `invoke` method
    accepting {"input": "<question>"} to keep compatibility with the rest of the app.
    The adapter will try several call patterns on the underlying chain to be
    tolerant of different versions of langchain-style chains.
    """
    qa = RetrievalQA(llm=llm, retriever=retriever)

    class QAChainAdapter:
        def __init__(self, qa_chain):
            self.qa_chain = qa_chain

        def invoke(self, params):
            # accept either {"input": "<question>"} or a plain string
            query = params.get("input") if isinstance(params, dict) else params

            # Try direct invoke if available
            if hasattr(self.qa_chain, "invoke"):
                try:
                    return self.qa_chain.invoke({"input": query})
                except Exception:
                    pass

            # Try run(...) which often returns a string answer
            if hasattr(self.qa_chain, "run"):
                try:
                    result = self.qa_chain.run(query)
                    return {"answer": result} if isinstance(result, str) else result
                except Exception:
                    pass

            # Try calling the chain as a callable
            try:
                result = self.qa_chain(query)
                return {"answer": result} if isinstance(result, str) else result
            except Exception as e:
                # Re-raise a clearer error for troubleshooting
                raise RuntimeError("Failed to invoke underlying QA chain") from e

    return QAChainAdapter(qa)

if st.session_state.get("vector_store") is None:
    st.info("Upload PDFs and build the index first to start asking questions.")
else:
    qcol1, qcol2 = st.columns([8,1])
    with qcol1:
        user_question = st.text_input("Type your question about the uploaded documents", key="user_question", label_visibility="collapsed")
    with qcol2:
        ask_btn = st.button("➡️ Ask")

    if st.session_state["is_generating"]:
        st.warning("⏳ Askly is generating an answer...")

    if ask_btn and user_question:
        st.session_state["is_generating"] = True
        with st.spinner("⏳ Generating answer..."):
            try:
                llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0)
                retriever = st.session_state["vector_store"].as_retriever()
                qa_chain = create_retrieval_chain(retriever, llm)

                t0 = time.perf_counter()
                response = qa_chain.invoke({"input": user_question})
                t1 = time.perf_counter()

                response_time = t1 - t0
                answer_str = response.get("answer") if isinstance(response, dict) else str(response)

                st.session_state["chat_history"].append({
                    "question": user_question,
                    "answer": answer_str,
                    "time": response_time
                })
                st.session_state["timings"].append(response_time)
            except Exception as e:
                st.error(f"Error generating answer: {e}")
            finally:
                st.session_state["is_generating"] = False
                # reset text input in UI
                try:
                    st.session_state.pop("user_question")
                except Exception:
                    pass
                st.rerun()

# ---------- Display chat history ----------
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

if st.session_state["chat_history"]:
    st.markdown("<div class='lm-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧾 Conversation</div>", unsafe_allow_html=True)

    for i, chat in enumerate(reversed(st.session_state["chat_history"])):
        # unique ids for copy widgets
        answer_id = f"lm-answer-{i}"
        copy_id = f"lm-copy-{i}"
        feedback_id = f"lm-feedback-{i}"
        st.markdown(f"""
        <div style='margin-bottom:12px; position:relative;'>
            <div class='lm-bubble-user'>{chat['question']}</div>
            <div style='height:8px;'></div>
            <div class='lm-bubble-assistant' id='{answer_id}'>{chat['answer']}</div>
            <button id="{copy_id}" style="position:absolute; right:6px; top:6px; background:transparent; border:none; color:#9be7ff; cursor:pointer;"
                onclick="
                    const t = document.getElementById('{answer_id}').innerText;
                    navigator.clipboard.writeText(t).then(() => {{
                        const fb = document.getElementById('{feedback_id}');
                        fb.style.opacity = 1;
                        setTimeout(() => {{ fb.style.opacity = 0; }}, 1600);
                    }});
                ">📋
            </button>
            <div id="{feedback_id}" class='copy-feedback' style='position:absolute; right:6px; top:32px;'>Copied!</div>
            <div style='margin-top:6px; font-size:12px; color:rgba(255,255,255,0.38);'>⏱ {chat['time']:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)

    # Performance summary (if available)
    if len(st.session_state["timings"]) >= 1:
        arr = np.array(st.session_state["timings"])
        st.markdown("---")
        st.markdown("### 📊 Performance Summary")
        st.write(
            f"- Average response: {arr.mean():.2f}s  \n"
            f"- Median response: {np.median(arr):.2f}s  \n"
            f"- 95th percentile: {np.percentile(arr,95):.2f}s"
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Askly · Dark Futuristic · Built with Streamlit and Groq LLM</div>", unsafe_allow_html=True)