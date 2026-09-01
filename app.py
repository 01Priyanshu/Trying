import streamlit as st
import google.generativeai as genai
import chromadb
from PIL import Image
import pypdf
import io
import base64
from streamlit_javascript import st_javascript

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="CSE 205 IA Assistant")

# Secret Handling
if "GEMINI_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_KEY"]
else:
    st.error("Missing GEMINI_KEY in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-pro')

# Custom Embedding Function (Fixed version)
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return [genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_document")["embedding"] for text in input]

# Initialize Vector DB
client = chromadb.PersistentClient(path="./cse205_data")
embedding_fn = GeminiEmbeddingFunction()
collection = client.get_or_create_collection(name="solutions", embedding_function=embedding_fn)

# --- 2. THE SCREEN CAPTURE ENGINE ---
def get_screen_image():
    # JavaScript to capture the screen and return it to Python
    js_code = """
    (async () => {
        const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
        const video = document.createElement('video');
        video.srcObject = stream;
        await video.play();
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL('image/png');
        // Stop the stream immediately after capture
        stream.getTracks().forEach(track => track.stop());
        return dataUrl;
    })()
    """
    return st_javascript(js_code)

# --- 3. SIDEBAR: UPLOAD PDFs ---
with st.sidebar:
    st.title("📚 Knowledge Base")
    st.write("Upload official course PDFs (Solutions/Activities) here.")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Index Documents"):
        if uploaded_files:
            with st.spinner("Indexing contents..."):
                for pdf in uploaded_files:
                    reader = pypdf.PdfReader(pdf)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            collection.upsert(
                                documents=[text],
                                metadatas=[{"source": pdf.name, "page": i+1}],
                                ids=[f"{pdf.name}_{i}"]
                            )
            st.success("Indexing Complete!")

# --- 4. MAIN INTERFACE ---
st.title("CSE 205 IA Assistant")
st.write("Click 'Analyze Screen' and select your Zoom window to diagnose student issues.")

if st.button("⚡ ANALYZE SCREEN", type="primary", use_container_width=True):
    raw_data = get_screen_image()
    
    if raw_data and len(raw_data) > 100:
        # Convert JS DataURL to PIL Image
        header, encoded = raw_data.split(",", 1)
        data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(data))
        
        # UI Display
        col_img, col_res = st.columns([1, 1.2])
        with col_img:
            st.image(image, caption="Captured Frame", use_column_width=True)

        with st.spinner("🤖 Searching PDFs and comparing code..."):
            # STEP 1: Identify Activity from Image
            id_prompt = "Identify the Activity Number or Lab Title from this screen (e.g., 'Activity 4.10.1')."
            id_res = model.generate_content([id_prompt, image])
            search_terms = id_res.text
            
            # STEP 2: Retrieve Official Solution from PDFs
            results = collection.query(query_texts=[search_terms], n_results=1)
            
            if not results['documents'][0]:
                st.error(f"Could not find a match for '{search_terms}' in your uploaded PDFs.")
            else:
                official_text = results['documents'][0][0]
                meta = results['metadatas'][0][0]
                
                # STEP 3: Diagnose against PDF
                final_prompt = f"""
                Analyze the student's code in the IMAGE vs the OFFICIAL SOLUTION below.
                
                OFFICIAL SOLUTION FROM PDF:
                {official_text}
                
                RULES:
                - Identify the EXACT error (discrepancy).
                - Do NOT guess. If not found, say 'Not Found'.
                - Reference {meta['source']} (Page {meta['page']}).

                SECTIONS:
                1. IA DIAGNOSIS: (Private fix)
                2. SAY TO STUDENT: (2-sentence hint)
                3. NEXT HINT: (Stronger clue)
                """
                
                analysis = model.generate_content([final_prompt, image])
                
                with col_res:
                    st.subheader("Results")
                    st.success(f"Matched: {search_terms}")
                    st.markdown(analysis.text)
                    
                    with st.expander("View Official Code"):
                        st.code(official_text, language="java")
    else:
        st.info("Please select a window in the browser popup to begin the analysis.")
