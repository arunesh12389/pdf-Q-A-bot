
# NoteMate – Your AI PDF Companion (Groq Edition)

NoteMate lets you upload a PDF (like lecture notes or manuals), ask questions, and get AI-powered answers based on the document content using Groq's LLMs.


## Tools Used
- **UI:** Streamlit
- **AI Model:** Groq API (e.g., Mixtral-8x7b-32768)
- **PDF Processing:** PyMuPDF (fitz)
- **Vector DB:** FAISS
- **LangChain:** For chaining workflow

## Setup
1. **Clone the repo**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set your Groq API key:**
   - Create a `.env` file in the root directory:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```
4. **Run the app:**
   ```bash
   streamlit run app.py
   ```


## Notes
- Large PDFs may take longer to process
- Only one PDF at a time is supported in this version

---
MIT License 
