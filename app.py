import streamlit as st
import google.generativeai as genai
import chromadb
from PIL import Image
import pypdf
import io
import base64
from streamlit_javascript import st_javascript

# --- 1. SETUP ---
st.set_page_config(layout="wide", page_title="CSE 205 IA Assistant (Live Link)")

if "GEMINI_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_KEY"]
else:
    st.error("Add GEMINI_KEY to Streamlit Secrets.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-pro')

# --- 2. CUSTOM EMBEDDING & DB ---
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return [genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_document")["embedding"] for text in input]

client = chromadb.PersistentClient(path="./cse205_kb")
embedding_fn = GeminiEmbeddingFunction()
collection = client.get_or_create_collection(name="solutions", embedding_function=embedding_fn)

# --- 3. JAVASCRIPT SCREEN CAPTURE COMPONENT ---
# This script allows the browser to capture a specific window (Zoom)
def capture_screen():
    # Simple JS to trigger browser screen picker
    js_code = """
    (async () => {
        if (!window.stream) {
            window.stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
        }
        const video = document.createElement('video');
        video.srcObject = window.stream;
        await video.play();
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL('image/png');
        return dataUrl;
    })()
    """
    return st_javascript(js_code)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("📚 PDF Knowledge Base")
    module_folder = st.selectbox("Module:", [f"Module {i}" for i in range(1, 15)] + ["Labs"])
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Index PDFs"):
        for pdf in uploaded_files:
            reader = pypdf.PdfReader(pdf)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    collection.upsert(documents=[text], metadatas=[{"source": pdf.name, "page": i+1}], ids=[f"{pdf.name}_{i}"])
        st.success("Indexed!")

# --- 5. MAIN LIVE INTERFACE ---
st.title("CSE 205 IA Assistant")
st.subheader("🔴 Live View Analysis")

col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.info("Step 1: Click 'Link Zoom Window' and select the student's share screen. \nStep 2: Click 'Sync & Analyze' whenever you need help.")
    
    # Capture Button
    img_data = capture_screen()
    
    if img_data and len(img_data) > 100:
        # Convert JS DataURL to Image for Gemini
        header, encoded = img_data.split(",", 1)
        data = base64.b64decode(encoded)
        current_frame = Image.open(io.BytesIO(data))
        st.image(current_frame, caption="Current Live View", use_column_width=True)
        
        analyze_btn = st.button("⚡ SYNC & ANALYZE NOW", type="primary", use_container_width=True)
    else:
        st.button("🔗 LINK ZOOM WINDOW", use_container_width=True)
        analyze_btn = False

# --- 6. AUTOMATED RAG LOGIC ---
if analyze_btn and current_frame:
    with st.spinner("Analyzing live screen against PDFs..."):
        # STEP 1: OCR identify from Live Frame
        id_res = model.generate_content(["Identify Activity/Lab # and method names from this screen.", current_frame])
        search_terms = id_res.text
        
        # STEP 2: RAG Search
        results = collection.query(query_texts=[search_terms], n_results=1)
        
        if not results['documents'][0]:
            st.error("Could not find matching PDF solution.")
        else:
            official_sol = results['documents'][0][0]
            meta = results['metadatas'][0][0]
            
            # STEP 3: Diagnose
            prompt = f"""
            Compare the code in the IMAGE to the OFFICIAL SOLUTION below.
            OFFICIAL SOLUTION: {official_sol}
            
            1. Find the exact logic/syntax error on the screen.
            2. Be direct. No 'maybe'.
            
            [IA DIAGNOSIS]
            Private fix for the IA. Reference {meta['source']} p.{meta['page']}.
            
            [SAY TO STUDENT]
            Natural 2- sentence hint.
            """
            
            analysis = model.generate_content([prompt, current_frame])
            
            with col_right:
                st.subheader("Analysis")
                st.markdown(analysis.text)
                if st.button("View Official Code"):
                    st.code(official_sol, language="java")
