import streamlit as st
import google.generativeai as genai
import chromadb
from PIL import Image
import pypdf
from datetime import datetime

# --- INITIAL CONFIGURATION ---
st.set_page_config(layout="wide", page_title="CSE 205 IA Assistant")

# API Key Setup
if "GEMINI_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_KEY"]
else:
    st.error("API Key not found in Secrets. Please add GEMINI_KEY to your Streamlit Cloud settings.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-pro')

# --- FIX: CUSTOM EMBEDDING FUNCTION ---
# This replaces the broken 'embedding_functions.GoogleGenerativeAiEmbeddingFunction'
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        model = "models/embedding-001"
        return [
            genai.embed_content(model=model, content=text, task_type="retrieval_document")["embedding"]
            for text in input
        ]

# --- IA ACCESS PASSWORD ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    if not st.session_state["password_correct"]:
        st.title("CSE 205 IA Assistant")
        pwd = st.text_input("Enter IA Access Code", type="password")
        if st.button("Login"):
            if pwd == "CSE205_IA_2024": 
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Access Denied.")
        return False
    return True

if not check_password():
    st.stop()

# --- DATABASE SETUP ---
# We now use our Custom GeminiEmbeddingFunction()
client = chromadb.PersistentClient(path="./cse205_db")
embedding_fn = GeminiEmbeddingFunction()
collection = client.get_or_create_collection(
    name="course_data", 
    embedding_function=embedding_fn
)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Admin: Knowledge Base")
    module = st.selectbox("Select Module", [f"Module {i}" for i in range(1, 13)] + ["Labs"])
    pdfs = st.file_uploader("Upload Official PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Index Files"):
        if pdfs:
            with st.spinner("Processing..."):
                for pdf in pdfs:
                    reader = pypdf.PdfReader(pdf)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            collection.upsert(
                                documents=[text],
                                metadatas=[{"module": module, "source": pdf.name, "page": i}],
                                ids=[f"{pdf.name}_{i}"]
                            )
            st.success("Indexing Complete.")
    
    if st.button("Clear Database"):
        client.delete_collection("course_data")
        st.rerun()

# --- MAIN INTERFACE ---
st.title("CSE 205 IA Assistant")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Analyze Student Work")
    screenshot = st.file_uploader("Upload Screenshot", type=['png', 'jpg', 'jpeg'])
    question = st.text_area("What is the student asking?")
    mode = st.radio("Mode", ["Individual Assignment", "Collaborative Lab"], horizontal=True)
    
    if st.button("ANALYZE", type="primary"):
        if not (screenshot or question):
            st.error("Please provide a screenshot or a question.")
        else:
            with st.spinner("Searching official solutions..."):
                # 1. Extract IDs from image
                context_str = ""
                if screenshot:
                    img = Image.open(screenshot)
                    ocr = model.generate_content(["Identify Activity/Lab number and method names from this image.", img])
                    context_str = ocr.text

                # 2. RAG Search
                search_query = f"{question} {context_str}"
                results = collection.query(query_texts=[search_query], n_results=2)

                if not results['documents'][0]:
                    with col2: st.error("Exact course solution not found. Please provide more context.")
                else:
                    # 3. Final Comparison
                    doc_text = "\n".join(results['documents'][0])
                    prompt = f"""
                    Role: CSE 205 Instructional Aide. 
                    Official Solution: {doc_text}
                    Student Question: {question}
                    Context: {context_str}
                    Mode: {mode}

                    Instructions:
                    - Diagnose the issue against the official solution.
                    - Provide a 'Say to Student' hint (2-5 sentences). 
                    - Do NOT reveal the official code.
                    - Be direct. No 'maybe' or 'probably'.

                    Format:
                    IA DIAGNOSIS: <private fix>
                    SAY TO STUDENT: <helpful hint>
                    NEXT HINT: <stronger hint>
                    """
                    
                    final_res = model.generate_content([prompt, img] if screenshot else [prompt])
                    
                    with col2:
                        st.subheader("IA Guidance")
                        st.markdown(final_res.text)
                        st.caption(f"Source: {results['metadatas'][0][0]['source']}")
